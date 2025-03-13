#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

__global__ void fill_random_with_eigs(cuComplex *d_data,
                                      const cuComplex *d_sqrt_eigs,
                                      int traj_size, int m,
                                      unsigned long seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= m * traj_size)
    return;

  int traj_id = tid / traj_size;
  int idx = tid % traj_size;

  __shared__ curandState state[32];
  int lane_id = threadIdx.x % 32;

  if (lane_id == 0) {
    curand_init(seed + traj_id, blockIdx.x, 0, &state[lane_id]);
  }
  __syncthreads();

  float re = curand_normal(&state[lane_id]);
  float im = curand_normal(&state);
  cuComplex noise = make_cuComplex(re, im);
  d_data[tid] = cuCmulf(noise, d_sqrt_eigs[idx]);
}

__global__ void scale_and_copy_to_output(const cuComplex *d_data,
                                         float *d_output, int n, int m,
                                         int offset, float scale) {
  int out_size = n - offset;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= m * out_size)
    return;

  int traj_id = tid / out_size;
  int idx = tid % out_size;
  int data_idx = traj_id * (2 * n) + (idx + 1);

  d_output[tid] = d_data[data_idx].x * scale;
}

extern "C" EXPORT void fgn_kernel(const cuComplex *d_sqrt_eigs, float *d_output,
                                  int n, int m, int offset, float hurst,
                                  float t, unsigned long seed) {
  int traj_size = 2 * n;
  cuComplex *d_data = nullptr;
  cudaMalloc(&d_data, (size_t)m * traj_size * sizeof(cuComplex));

  int block_size = 512;
  int grid_size = (m * traj_size + block_size - 1) / block_size;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  fill_random_with_eigs<<<gridSize, blockSize, 0, stream>>>(d_data, d_sqrt_eigs,
                                                            traj_size, m, seed);

  cufftHandle plan;
  cufftPlan1d(&plan, traj_size, CUFFT_C2C, m);
  cufftSetStream(plan, stream);
  cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
  cufftDestroy(plan);

  int out_size = n - offset;
  grid_size = (m * out_size + block_size - 1) / block_size;
  float scale = powf((float)n, -hurst) * powf(t, hurst);
  scale_and_copy_to_output<<<gridSize, blockSize, 0, stream>>>(
      d_data, d_output, n, m, offset, scale);

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  cudaFree(d_data);
}
