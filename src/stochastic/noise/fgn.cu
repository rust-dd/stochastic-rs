#include "fgn.cuh"
#include <cuComplex.h>
#include <curand_kernel.h>
#include <cufft.h>
#include <math.h>


__global__ void r_kernel(float* r, int n, float hurst)
{

}

__global__ void sqrt_eigenvalues_kernel(cufftComplex* r_fft, cuComplex* sqrt_eigenvalues, int n)
{

}

extern "C" void sqrt_eigenvalues_kernel_wrapper(cuComplex* sqrt_eigenvalues, int n, int hurst)
{

}

__global__ void fgn_kernel(const cuComplex* sqrt_eigenvalues, cuComplex* result, int n, float scale, unsigned long seed)
{

}

extern "C" void fgn_kernel_wrapper(const cuComplex* sqrt_eigenvalues, cuComplex* result, int n, int m, float scale, unsigned long seed)
{

}
