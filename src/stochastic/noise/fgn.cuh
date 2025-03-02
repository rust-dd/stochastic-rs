#pragma once
#include <cuComplex.h>

#ifdef __cplusplus
extern "C" {
#endif

void sqrt_eigenvalues_kernel_wrapper(
    cuComplex* sqrt_eigenvalues,
    int n,
    float hurst
);

void fgn_kernel_wrapper(
    const cuComplex* sqrt_eigenvalues,
    cuComplex* results,
    int n,
    int m,
    float scale,
    unsigned long seed
);

#ifdef __cplusplus
}
#endif
