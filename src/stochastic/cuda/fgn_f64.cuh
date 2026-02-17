#pragma once

#include "fgn_common.cuh"

extern "C" EXPORT void fgn64_cleanup();

__global__ void fill_random_with_eigs_fast_f64(
    cuDoubleComplex *__restrict__ d_data,
    const cuDoubleComplex *__restrict__ d_sqrt_eigs,
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
  double2 normal = curand_normal2_double(&state);
  double eig_re = d_sqrt_eigs[idx].x;
  d_data[tid] = make_cuDoubleComplex(normal.x * eig_re, normal.y * eig_re);
}

__global__ void scale_and_copy_to_output_f64(
    const cuDoubleComplex *__restrict__ d_data,
    double *__restrict__ d_output,
    int out_size,
    int traj_stride,
    double scale,
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

static cuDoubleComplex *g64_sqrt_eigs = nullptr;
static cuDoubleComplex *g64_data = nullptr;
static double *g64_output = nullptr;
static cufftHandle g64_plan = 0;
static cudaStream_t g64_stream = nullptr;
static int g64_n = 0;
static int g64_m = 0;
static int g64_traj_size = 0;
static int g64_out_size = 0;

extern "C" EXPORT int fgn64_init(
    const cuDoubleComplex *h_sqrt_eigs,
    int eig_len,
    int n,
    int m,
    int offset)
{
  fgn64_cleanup();

  if (n <= 0 || m <= 0 || eig_len <= 0) {
    return 1;
  }

  g64_n = n;
  g64_m = m;
  g64_traj_size = 2 * n;
  g64_out_size = n - offset;

  cudaMalloc(&g64_sqrt_eigs, (size_t)eig_len * sizeof(cuDoubleComplex));
  cudaMemcpy(g64_sqrt_eigs, h_sqrt_eigs, (size_t)eig_len * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  cudaMalloc(&g64_data, (size_t)m * (size_t)g64_traj_size * sizeof(cuDoubleComplex));
  cudaMalloc(&g64_output, (size_t)m * (size_t)g64_out_size * sizeof(double));

  if (cufftPlan1d(&g64_plan, g64_traj_size, CUFFT_Z2Z, m) != CUFFT_SUCCESS) {
    fgn64_cleanup();
    return 2;
  }

  cudaStreamCreate(&g64_stream);
  cufftSetStream(g64_plan, g64_stream);
  return 0;
}

extern "C" EXPORT void fgn64_sample(
    double *h_output,
    double scale,
    unsigned long long seed)
{
  if (!g64_data || !g64_output || !g64_plan) {
    return;
  }

  int total_data = g64_m * g64_traj_size;
  int total_out = g64_m * g64_out_size;
  int block_size = 256;

  int grid_data = (total_data + block_size - 1) / block_size;
  fill_random_with_eigs_fast_f64<<<grid_data, block_size, 0, g64_stream>>>(
      g64_data, g64_sqrt_eigs, g64_traj_size, total_data, seed);

  cufftExecZ2Z(g64_plan, (cufftDoubleComplex *)g64_data, (cufftDoubleComplex *)g64_data, CUFFT_FORWARD);

  int grid_out = (total_out + block_size - 1) / block_size;
  scale_and_copy_to_output_f64<<<grid_out, block_size, 0, g64_stream>>>(
      g64_data, g64_output, g64_out_size, g64_traj_size, scale, total_out);

  cudaMemcpyAsync(h_output, g64_output, (size_t)total_out * sizeof(double), cudaMemcpyDeviceToHost, g64_stream);
  cudaStreamSynchronize(g64_stream);
}

extern "C" EXPORT void fgn64_cleanup()
{
  if (g64_plan) {
    cufftDestroy(g64_plan);
    g64_plan = 0;
  }
  if (g64_stream) {
    cudaStreamDestroy(g64_stream);
    g64_stream = nullptr;
  }
  if (g64_sqrt_eigs) {
    cudaFree(g64_sqrt_eigs);
    g64_sqrt_eigs = nullptr;
  }
  if (g64_data) {
    cudaFree(g64_data);
    g64_data = nullptr;
  }
  if (g64_output) {
    cudaFree(g64_output);
    g64_output = nullptr;
  }
}
