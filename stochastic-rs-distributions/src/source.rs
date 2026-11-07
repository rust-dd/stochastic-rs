//! Scalar draw sources: the SIMD engines for the kernels' slow paths, any `rand` rng for the honest draws.

use rand::Rng;
use stochastic_rs_core::simd_rng::SimdRngExt;

use crate::normal::SimdNormal;
use crate::seeded::StreamState;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::SimdKernel;

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

/// The standard normal and the `[0, 1)` uniform a scalar rejection step consumes, from a stream's parts or the
/// caller's rng.
pub(crate) trait NormalUniformSource<T> {
  fn normal(&mut self) -> T;

  fn uniform(&mut self) -> T;
}

impl<T: SimdFloatExt, R: SimdRngExt> NormalUniformSource<T>
  for (&mut StreamState<T, R, 64>, &mut R)
{
  #[inline]
  fn normal(&mut self) -> T {
    SimdNormal::<T>::standard().next(self.0)
  }

  #[inline]
  fn uniform(&mut self) -> T {
    T::sample_uniform_simd(self.1)
  }
}

impl<T: SimdFloatExt, G: Rng + ?Sized> NormalUniformSource<T> for AnyRng<'_, G> {
  /// Kept out of line because inlined into the forced-inline gamma trial this scalar ziggurat slows the honest draw.
  #[inline(never)]
  fn normal(&mut self) -> T {
    SimdNormal::<T>::standard().draw_with(self.0)
  }

  #[inline]
  fn uniform(&mut self) -> T {
    T::sample_uniform(self.0)
  }
}
