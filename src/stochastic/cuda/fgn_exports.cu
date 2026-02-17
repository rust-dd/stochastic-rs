#include "fgn_common.cuh"
#include "fgn_f32.cuh"
#include "fgn_f64.cuh"

extern "C" EXPORT int fgn_init(
    const cuComplex *h_sqrt_eigs,
    int eig_len,
    int n,
    int m,
    int offset)
{
  return fgn32_init(h_sqrt_eigs, eig_len, n, m, offset);
}

extern "C" EXPORT void fgn_sample(
    float *h_output,
    float scale,
    unsigned long long seed)
{
  fgn32_sample(h_output, scale, seed);
}

extern "C" EXPORT void fgn_cleanup()
{
  fgn32_cleanup();
}
