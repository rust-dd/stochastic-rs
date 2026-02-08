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

__global__ void fill_random_with_eigs_fast(
    cuComplex *__restrict__ d_data,
    const cuComplex *__restrict__ d_sqrt_eigs,
    int traj_size,
    int total,
    unsigned long seed)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total)
    return;

  int idx = tid % traj_size;

  // Use curand's Philox4_32_10 state for high-quality random numbers
  curandStatePhilox4_32_10_t state;
  curand_init(seed, tid, 0, &state);

  // Generate two standard normal random numbers
  float2 normal = curand_normal2(&state);

  // Load eigenvalue (eigenvalues are real, imaginary part is ~0)
  float eig_re = d_sqrt_eigs[idx].x;
  // Complex multiplication: (normal.x + i*normal.y) * eig_re
  d_data[tid] = make_cuComplex(normal.x * eig_re, normal.y * eig_re);
}

__global__ void scale_and_copy_to_output(
    const cuComplex *__restrict__ d_data,
    float *__restrict__ d_output,
    int n,
    int out_size,
    int traj_stride,
    float scale,
    int total)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total)
    return;
  int traj_id = tid / out_size;
  int idx = tid % out_size;
  int data_idx = traj_id * traj_stride + (idx + 1);
  d_output[tid] = d_data[data_idx].x * scale;
}

extern "C" EXPORT void fgn_kernel(
    const cuComplex *d_sqrt_eigs,
    float *d_output,
    int n,
    int m,
    int offset,
    float scale,
    unsigned long seed)
{
  int traj_size = 2 * n;
  int out_size = n - offset;
  int total_data = m * traj_size;
  int total_out = m * out_size;

  // Allocate working buffer
  cuComplex *d_data = nullptr;
  cudaMalloc(&d_data, (size_t)total_data * sizeof(cuComplex));

  // Create stream for async execution
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Fill with random data * eigenvalues (optimized kernel)
  {
    int blockSize = 256;
    int gridSize = (total_data + blockSize - 1) / blockSize;
    fill_random_with_eigs_fast<<<gridSize, blockSize, 0, stream>>>(d_data, d_sqrt_eigs, traj_size, total_data, seed);
  }

  // Batched FFT
  {
    cufftHandle plan;
    cufftPlan1d(&plan, traj_size, CUFFT_C2C, m);
    cufftSetStream(plan, stream);
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftDestroy(plan);
  }

  // Scale and copy output
  {
    int blockSize = 256;
    int gridSize = (total_out + blockSize - 1) / blockSize;
    scale_and_copy_to_output<<<gridSize, blockSize, 0, stream>>>(d_data, d_output, n, out_size, traj_size, scale, total_out);
  }

  // Single sync at the end
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  cudaFree(d_data);
}

// Persistent context for repeated sampling
static cuComplex *g_sqrt_eigs = nullptr;
static cuComplex *g_data = nullptr;
static float *g_output = nullptr;
static cufftHandle g_plan = 0;
static cudaStream_t g_stream = nullptr;
static int g_n = 0;
static int g_m = 0;
static int g_traj_size = 0;
static int g_out_size = 0;

extern "C" EXPORT int fgn_init(
    const cuComplex *h_sqrt_eigs,
    int eig_len,
    int n,
    int m,
    int offset)
{
  g_n = n;
  g_m = m;
  g_traj_size = 2 * n;
  g_out_size = n - offset;

  // Allocate persistent buffers
  cudaMalloc(&g_sqrt_eigs, eig_len * sizeof(cuComplex));
  cudaMemcpy(g_sqrt_eigs, h_sqrt_eigs, eig_len * sizeof(cuComplex), cudaMemcpyHostToDevice);

  cudaMalloc(&g_data, (size_t)m * g_traj_size * sizeof(cuComplex));
  cudaMalloc(&g_output, (size_t)m * g_out_size * sizeof(float));

  // Create persistent FFT plan
  cufftPlan1d(&g_plan, g_traj_size, CUFFT_C2C, m);

  // Create persistent stream
  cudaStreamCreate(&g_stream);
  cufftSetStream(g_plan, g_stream);

  return 0;
}

extern "C" EXPORT void fgn_sample(
    float *h_output,
    float scale,
    unsigned long seed)
{
  int total_data = g_m * g_traj_size;
  int total_out = g_m * g_out_size;

  // Fill with random data * eigenvalues
  {
    int blockSize = 256;
    int gridSize = (total_data + blockSize - 1) / blockSize;
    fill_random_with_eigs_fast<<<gridSize, blockSize, 0, g_stream>>>(g_data, g_sqrt_eigs, g_traj_size, total_data, seed);
  }

  // Batched FFT (plan already set with stream)
  cufftExecC2C(g_plan, g_data, g_data, CUFFT_FORWARD);

  // Scale and copy output
  {
    int blockSize = 256;
    int gridSize = (total_out + blockSize - 1) / blockSize;
    scale_and_copy_to_output<<<gridSize, blockSize, 0, g_stream>>>(g_data, g_output, g_n, g_out_size, g_traj_size, scale, total_out);
  }

  // Copy result back to host
  cudaMemcpyAsync(h_output, g_output, total_out * sizeof(float), cudaMemcpyDeviceToHost, g_stream);
  cudaStreamSynchronize(g_stream);
}

extern "C" EXPORT void fgn_cleanup()
{
  if (g_plan) {
    cufftDestroy(g_plan);
    g_plan = 0;
  }
  if (g_stream) {
    cudaStreamDestroy(g_stream);
    g_stream = nullptr;
  }
  if (g_sqrt_eigs) {
    cudaFree(g_sqrt_eigs);
    g_sqrt_eigs = nullptr;
  }
  if (g_data) {
    cudaFree(g_data);
    g_data = nullptr;
  }
  if (g_output) {
    cudaFree(g_output);
    g_output = nullptr;
  }
}
