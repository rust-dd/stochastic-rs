#include <curand_kernel.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>

extern "C" __global__ void fgn_kernel(
    const cuComplex* sqrt_eigenvalues,
    float* output,
    int n,
    int m,
    int offset,
    float hurst,
    float t,
    unsigned long seed
) {
    int traj_idx = blockIdx.x;
    if (traj_idx >= m) return;

    int idx = threadIdx.x;
    int traj_size = 2 * n;
    int output_size = n - offset;

    extern __shared__ cuComplex shared_data[];

    if (idx < traj_size) {
        curandState state;
        curand_init(seed + traj_idx, idx, 0, &state);

        float real = curand_normal(&state);
        float imag = curand_normal(&state);
        cuComplex noise = make_cuComplex(real, imag);
        shared_data[idx] = cuCmulf(noise, sqrt_eigenvalues[idx]);
    }

    __syncthreads();

    if (idx == 0) {
        cufftHandle plan;
        cufftComplex* data = (cufftComplex*)shared_data;
        // TODO: need to optimize, because create a plan for FFT in every thread not efficient
        cufftPlan1d(&plan, traj_size, CUFFT_C2C, 1);
        cufftExecC2C(plan, data, data, CUFFT_FORWARD);
        cufftDestroy(plan);
    }

    __syncthreads();

    float scale = powf((float)n, -hurst) * powf(t, hurst);
    if (idx < output_size) {
        int output_offset = traj_idx * output_size;
        output[output_offset + idx] = shared_data[idx + 1].x * scale;
    }
}
