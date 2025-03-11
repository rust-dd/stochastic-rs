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
    cuComplex* d_data,
    const cuComplex* d_sqrt_eigs,
    int traj_size,
    int m,
    unsigned long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= m * traj_size) return;
    int traj_id = tid / traj_size;
    int idx = tid % traj_size;
    curandState state;
    curand_init(seed + traj_id, idx, 0, &state);
    float re = curand_normal(&state);
    float im = curand_normal(&state);
    cuComplex noise = make_cuComplex(re, im);
    d_data[tid] = cuCmulf(noise, d_sqrt_eigs[idx]);
}

__global__ void scale_and_copy_to_output(
    const cuComplex* d_data,
    float* d_output,
    int n,
    int m,
    int offset,
    float hurst,
    float t)
{
    int out_size = n - offset;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= m * out_size) return;
    int traj_id = tid / out_size;
    int idx = tid % out_size;
    int data_idx = traj_id * (2 * n) + (idx + 1);
    float scale = powf((float)n, -hurst) * powf(t, hurst);
    d_output[tid] = d_data[data_idx].x * scale;
}

extern "C" EXPORT void fgn_kernel(
    const cuComplex* d_sqrt_eigs,
    float* d_output,
    int n,
    int m,
    int offset,
    float hurst,
    float t,
    unsigned long seed)
{
    int traj_size = 2 * n;
    cuComplex* d_data = nullptr;
    cudaMalloc(&d_data, (size_t)m * traj_size * sizeof(cuComplex));
    {
        int totalThreads = m * traj_size;
        int blockSize = 512;
        int gridSize = (totalThreads + blockSize - 1) / blockSize;
        fill_random_with_eigs<<<gridSize, blockSize>>>(d_data, d_sqrt_eigs, traj_size, m, seed);
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
        int out_size = n - offset;
        int totalThreads = m * out_size;
        int blockSize = 512;
        int gridSize = (totalThreads + blockSize - 1) / blockSize;
        scale_and_copy_to_output<<<gridSize, blockSize>>>(d_data, d_output, n, m, offset, hurst, t);
        cudaDeviceSynchronize();
    }
    cudaFree(d_data);
}
