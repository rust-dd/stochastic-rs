#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cufft.h>
#include <cuComplex.h>
#include <math.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

__global__ void fill_random_with_eigs(
    cuComplex *d_data,
    const cuComplex *d_sqrt_eigs,
    int traj_size,
    int m,
    unsigned long seed)
{
    int traj_id = blockIdx.x;
    if (traj_id >= m)
        return;

    int idx = threadIdx.x;
    if (idx >= traj_size)
        return;

    int data_idx = traj_id * traj_size + idx;

    curandState state;
    curand_init(seed + traj_id, idx, 0, &state);

    float re = curand_normal(&state);
    float im = curand_normal(&state);
    cuComplex noise = make_cuComplex(re, im);

    d_data[data_idx] = cuCmulf(noise, d_sqrt_eigs[idx]);
}

__global__ void scale_and_copy_to_output(
    const cuComplex *d_data,
    float *d_output,
    int n,
    int m,
    int offset,
    float hurst,
    float t)
{
    int traj_id = blockIdx.x;
    if (traj_id >= m)
        return;

    int idx = threadIdx.x;
    int out_size = n - offset;
    if (idx >= out_size)
        return;

    int data_idx = traj_id * (2 * n) + (idx + 1);
    float scale = powf((float)n, -hurst) * powf(t, hurst);

    int out_idx = traj_id * out_size + idx;
    d_output[out_idx] = d_data[data_idx].x * scale;
}

extern "C" EXPORT void fgn_kernel(
    const cuComplex *d_sqrt_eigs,
    float *d_output,
    int n,
    int m,
    int offset,
    float hurst,
    float t,
    unsigned long seed)
{
    int traj_size = 2 * n;

    cuComplex *d_data = nullptr;
    cudaMalloc(&d_data, (size_t)m * traj_size * sizeof(cuComplex));

    {
        dim3 gridDim(m);
        dim3 blockDim(traj_size);
        fill_random_with_eigs<<<gridDim, blockDim>>>(
            d_data, d_sqrt_eigs, traj_size, m, seed);
        cudaDeviceSynchronize();
    }

    {
        cufftHandle plan;
        cufftPlan1d(&plan, traj_size, CUFFT_C2C, m);
        cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
        cudaDeviceSynchronize();
        cufftDestroy(plan);
    }

    {
        dim3 gridDim(m);
        dim3 blockDim(n);
        scale_and_copy_to_output<<<gridDim, blockDim>>>(
            d_data, d_output, n, m, offset, hurst, t);
        cudaDeviceSynchronize();
    }

    cudaFree(d_data);
}
