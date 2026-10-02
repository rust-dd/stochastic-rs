//! Scalar draw sources: the SIMD engines for the kernels' slow paths, any `rand` rng for the honest draws.

use rand::Rng;
use stochastic_rs_core::simd_rng::SimdRngExt;

/// The two scalar draws a slow path needs, from a SIMD engine or from any `rand` rng.
pub(crate) trait Source {
  fn next_i32(&mut self) -> i32;

  fn next_f64(&mut self) -> f64;
}

impl<R: SimdRngExt> Source for R {
  #[inline(always)]
  fn next_i32(&mut self) -> i32 {
    SimdRngExt::next_i32(self)
  }

  #[inline(always)]
  fn next_f64(&mut self) -> f64 {
    SimdRngExt::next_f64(self)
  }
}

/// Adapter over the caller's rng: 53-bit uniforms from `next_u64` alone.
pub(crate) struct AnyRng<'a, G: Rng + ?Sized>(pub &'a mut G);

impl<G: Rng + ?Sized> Source for AnyRng<'_, G> {
  #[inline(always)]
  fn next_i32(&mut self) -> i32 {
    self.0.next_u32() as i32
  }

  #[inline(always)]
  fn next_f64(&mut self) -> f64 {
    uniform53(self.0.next_u64())
  }
}

/// `(bits >> 11) · 2⁻⁵³`, bit-identical to rand's `StandardUniform` for `f64`.
#[inline(always)]
pub(crate) fn uniform53(bits: u64) -> f64 {
  (bits >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}
