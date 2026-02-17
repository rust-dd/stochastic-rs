#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cufft.h>
#include <cuComplex.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
