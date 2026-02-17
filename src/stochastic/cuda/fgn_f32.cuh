#pragma once

#include "fgn_common.cuh"

extern "C" EXPORT void fgn32_cleanup();

__global__ void fill_random_with_eigs_fast_f32(
    cuComplex *__restrict__ d_data,
    const cuComplex *__restrict__ d_sqrt_eigs,
    int traj_size,
    int total,
    unsigned long long seed)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total) {
    return;
  }

  int idx = tid % traj_size;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, (unsigned long long)tid, 0, &state);
  float2 normal = curand_normal2(&state);
  float eig_re = d_sqrt_eigs[idx].x;
  d_data[tid] = make_cuComplex(normal.x * eig_re, normal.y * eig_re);
}

__global__ void scale_and_copy_to_output_f32(
    const cuComplex *__restrict__ d_data,
    float *__restrict__ d_output,
    int out_size,
    int traj_stride,
    float scale,
    int total)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total) {
    return;
  }
  int traj_id = tid / out_size;
  int idx = tid % out_size;
  int data_idx = traj_id * traj_stride + (idx + 1);
  d_output[tid] = d_data[data_idx].x * scale;
}

static cuComplex *g32_sqrt_eigs = nullptr;
static cuComplex *g32_data = nullptr;
static float *g32_output = nullptr;
static cufftHandle g32_plan = 0;
static cudaStream_t g32_stream = nullptr;
static int g32_n = 0;
static int g32_m = 0;
static int g32_traj_size = 0;
static int g32_out_size = 0;

extern "C" EXPORT int fgn32_init(
    const cuComplex *h_sqrt_eigs,
    int eig_len,
    int n,
    int m,
    int offset)
{
  fgn32_cleanup();

  if (n <= 0 || m <= 0 || eig_len <= 0) {
    return 1;
  }

  g32_n = n;
  g32_m = m;
  g32_traj_size = 2 * n;
  g32_out_size = n - offset;

  cudaMalloc(&g32_sqrt_eigs, (size_t)eig_len * sizeof(cuComplex));
  cudaMemcpy(g32_sqrt_eigs, h_sqrt_eigs, (size_t)eig_len * sizeof(cuComplex), cudaMemcpyHostToDevice);
  cudaMalloc(&g32_data, (size_t)m * (size_t)g32_traj_size * sizeof(cuComplex));
  cudaMalloc(&g32_output, (size_t)m * (size_t)g32_out_size * sizeof(float));

  if (cufftPlan1d(&g32_plan, g32_traj_size, CUFFT_C2C, m) != CUFFT_SUCCESS) {
    fgn32_cleanup();
    return 2;
  }

  cudaStreamCreate(&g32_stream);
  cufftSetStream(g32_plan, g32_stream);
  return 0;
}

extern "C" EXPORT void fgn32_sample(
    float *h_output,
    float scale,
    unsigned long long seed)
{
  if (!g32_data || !g32_output || !g32_plan) {
    return;
  }

  int total_data = g32_m * g32_traj_size;
  int total_out = g32_m * g32_out_size;
  int block_size = 256;

  int grid_data = (total_data + block_size - 1) / block_size;
  fill_random_with_eigs_fast_f32<<<grid_data, block_size, 0, g32_stream>>>(
      g32_data, g32_sqrt_eigs, g32_traj_size, total_data, seed);

  cufftExecC2C(g32_plan, (cufftComplex *)g32_data, (cufftComplex *)g32_data, CUFFT_FORWARD);

  int grid_out = (total_out + block_size - 1) / block_size;
  scale_and_copy_to_output_f32<<<grid_out, block_size, 0, g32_stream>>>(
      g32_data, g32_output, g32_out_size, g32_traj_size, scale, total_out);

  cudaMemcpyAsync(h_output, g32_output, (size_t)total_out * sizeof(float), cudaMemcpyDeviceToHost, g32_stream);
  cudaStreamSynchronize(g32_stream);
}

extern "C" EXPORT void fgn32_cleanup()
{
  if (g32_plan) {
    cufftDestroy(g32_plan);
    g32_plan = 0;
  }
  if (g32_stream) {
    cudaStreamDestroy(g32_stream);
    g32_stream = nullptr;
  }
  if (g32_sqrt_eigs) {
    cudaFree(g32_sqrt_eigs);
    g32_sqrt_eigs = nullptr;
  }
  if (g32_data) {
    cudaFree(g32_data);
    g32_data = nullptr;
  }
  if (g32_output) {
    cudaFree(g32_output);
    g32_output = nullptr;
  }
}
