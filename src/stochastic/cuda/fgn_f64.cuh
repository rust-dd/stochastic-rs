#pragma once

#include "fgn_common.cuh"
#include <chrono>
#include <cstring>

extern "C" EXPORT void fgn64_cleanup();

__global__ void init_rng_states_f64(
    curandStatePhilox4_32_10_t *__restrict__ states,
    int total,
    unsigned long long seed)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total) {
    return;
  }
  curand_init(seed, (unsigned long long)tid, 0, &states[tid]);
}

__global__ void fill_random_with_eigs_fast_f64(
    cuDoubleComplex *__restrict__ d_data,
    curandStatePhilox4_32_10_t *__restrict__ states,
    const cuDoubleComplex *__restrict__ d_sqrt_eigs,
    int traj_size,
    int total)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total) {
    return;
  }

  int idx = tid % traj_size;
  curandStatePhilox4_32_10_t state = states[tid];
  double2 normal = curand_normal2_double(&state);
  states[tid] = state;
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
static curandStatePhilox4_32_10_t *g64_states = nullptr;
static double *g64_host_output = nullptr;
static size_t g64_host_output_bytes = 0;
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

  int total_data = g64_m * g64_traj_size;
  g64_host_output_bytes = (size_t)m * (size_t)g64_out_size * sizeof(double);

  cudaMalloc(&g64_sqrt_eigs, (size_t)eig_len * sizeof(cuDoubleComplex));
  cudaMemcpy(g64_sqrt_eigs, h_sqrt_eigs, (size_t)eig_len * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  cudaMalloc(&g64_data, (size_t)m * (size_t)g64_traj_size * sizeof(cuDoubleComplex));
  cudaMalloc(&g64_output, (size_t)m * (size_t)g64_out_size * sizeof(double));
  cudaMalloc(&g64_states, (size_t)total_data * sizeof(curandStatePhilox4_32_10_t));
  if (cudaHostAlloc((void **)&g64_host_output, g64_host_output_bytes, cudaHostAllocDefault) != cudaSuccess) {
    fgn64_cleanup();
    return 5;
  }

  if (cufftPlan1d(&g64_plan, g64_traj_size, CUFFT_Z2Z, m) != CUFFT_SUCCESS) {
    fgn64_cleanup();
    return 2;
  }

  cudaStreamCreate(&g64_stream);
  cufftSetStream(g64_plan, g64_stream);

  int block_size = 256;
  int grid_states = (total_data + block_size - 1) / block_size;
  unsigned long long init_seed = (unsigned long long)
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  init_seed ^= 0x9e3779b97f4a7c15ULL;
  init_seed ^= (unsigned long long)(g64_n * 1315423911u);
  init_seed ^= (unsigned long long)g64_m;
  init_rng_states_f64<<<grid_states, block_size, 0, g64_stream>>>(
      g64_states, total_data, init_seed);
  if (cudaGetLastError() != cudaSuccess) {
    fgn64_cleanup();
    return 3;
  }
  if (cudaStreamSynchronize(g64_stream) != cudaSuccess) {
    fgn64_cleanup();
    return 4;
  }
  return 0;
}

extern "C" EXPORT void fgn64_sample(
    double *h_output,
    double scale,
    unsigned long long seed)
{
  (void)seed;
  if (!g64_data || !g64_output || !g64_plan || !g64_host_output) {
    return;
  }

  int total_data = g64_m * g64_traj_size;
  int total_out = g64_m * g64_out_size;
  int block_size = 256;

  int grid_data = (total_data + block_size - 1) / block_size;
  fill_random_with_eigs_fast_f64<<<grid_data, block_size, 0, g64_stream>>>(
      g64_data, g64_states, g64_sqrt_eigs, g64_traj_size, total_data);

  cufftExecZ2Z(g64_plan, (cufftDoubleComplex *)g64_data, (cufftDoubleComplex *)g64_data, CUFFT_FORWARD);

  int grid_out = (total_out + block_size - 1) / block_size;
  scale_and_copy_to_output_f64<<<grid_out, block_size, 0, g64_stream>>>(
      g64_data, g64_output, g64_out_size, g64_traj_size, scale, total_out);

  cudaMemcpyAsync(g64_host_output, g64_output, g64_host_output_bytes, cudaMemcpyDeviceToHost, g64_stream);
  cudaStreamSynchronize(g64_stream);
  std::memcpy(h_output, g64_host_output, g64_host_output_bytes);
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
  if (g64_states) {
    cudaFree(g64_states);
    g64_states = nullptr;
  }
  if (g64_host_output) {
    cudaFreeHost(g64_host_output);
    g64_host_output = nullptr;
  }
  g64_host_output_bytes = 0;
}
