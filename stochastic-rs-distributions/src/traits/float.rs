//! Foundational `RealExt`, `SimdFloatExt` and `FloatExt` traits.

use std::fmt::Debug;
use std::iter::Sum;
use std::ops::AddAssign;
use std::ops::SubAssign;

use ndarray::Array1;
use ndarray::ScalarOperand;
use num_complex::Complex;
use rand::Rng;

/// Scalar real-number surface: arithmetic, conversions and constants —
/// nothing that draws randomness and nothing that requires SIMD lanes.
///
/// This is the bound for code that only *computes*: analytic pricers,
/// characteristic functions, Greeks, calibration objectives, quadrature.
/// Because it asks for no lane type and no RNG, a custom scalar — an AAD
/// dual number, a tape node, a higher-precision float — can implement it,
/// which is exactly what [`FloatExt`] forbids: that trait additionally
/// demands the 8-lane SIMD surface and RNG-backed fills, so `f32`/`f64`
/// are its only possible implementors.
pub trait RealExt:
  num_traits::Float
  + num_traits::FromPrimitive
  + num_traits::Signed
  + num_traits::FloatConst
  + Sum
  + num_traits::Zero
  + Default
  + Debug
  + Send
  + Sync
  + ScalarOperand
  + AddAssign
  + SubAssign
  + 'static
{
  fn from_usize_(n: usize) -> Self;
  /// Cheap `f64 -> Self` conversion for compile-time-known constants; a
  /// plain `as`-style cast rather than the checked `FromPrimitive` route.
  fn from_f64_fast(v: f64) -> Self;
  #[inline(always)]
  fn from_f32_fast(v: f32) -> Self {
    Self::from_f64_fast(v as f64)
  }

  fn pi() -> Self;
  fn two_pi() -> Self;
  fn min_positive_val() -> Self;
}

/// 8-lane SIMD surface over a [`RealExt`] scalar, plus the uniform RNG
/// fills the SIMD samplers draw from.
pub trait SimdFloatExt: RealExt {
  type Simd: Copy
    + std::ops::Mul<Output = Self::Simd>
    + std::ops::Add<Output = Self::Simd>
    + std::ops::Sub<Output = Self::Simd>
    + std::ops::Div<Output = Self::Simd>
    + std::ops::Neg<Output = Self::Simd>;

  fn splat(val: Self) -> Self::Simd;
  fn simd_from_array(arr: [Self; 8]) -> Self::Simd;
  fn simd_to_array(v: Self::Simd) -> [Self; 8];
  fn simd_ln(v: Self::Simd) -> Self::Simd;
  fn simd_sqrt(v: Self::Simd) -> Self::Simd;
  fn simd_cos(v: Self::Simd) -> Self::Simd;
  fn simd_sin(v: Self::Simd) -> Self::Simd;
  fn simd_exp(v: Self::Simd) -> Self::Simd;
  fn simd_tan(v: Self::Simd) -> Self::Simd;
  fn simd_max(a: Self::Simd, b: Self::Simd) -> Self::Simd;
  fn simd_powf(v: Self::Simd, exp: Self) -> Self::Simd;
  fn fill_uniform<R: Rng + ?Sized>(rng: &mut R, out: &mut [Self]);
  fn fill_uniform_simd<R: crate::simd_rng::SimdRngExt>(rng: &mut R, out: &mut [Self]);
  fn sample_uniform<R: Rng + ?Sized>(rng: &mut R) -> Self;
  #[inline(always)]
  fn sample_uniform_simd<R: crate::simd_rng::SimdRngExt>(rng: &mut R) -> Self {
    let mut buf = [Self::zero(); 8];
    Self::fill_uniform_simd(rng, &mut buf);
    buf[0]
  }

  fn simd_from_i32x8(v: wide::i32x8) -> Self::Simd;
  const PREFERS_F32_WN: bool = false;
}

/// The full simulation-grade float: [`RealExt`] scalar arithmetic,
/// [`SimdFloatExt`] lanes, and the batched standard-normal / fGN scratch
/// machinery every path sampler draws through.
pub trait FloatExt: RealExt + SimdFloatExt {
  fn fill_standard_normal_slice(out: &mut [Self]);
  #[inline]
  fn fill_standard_normal_scaled_slice(out: &mut [Self], scale: Self) {
    Self::fill_standard_normal_slice(out);
    for x in out.iter_mut() {
      *x = *x * scale;
    }
  }

  fn with_fgn_complex_scratch<R, F: FnOnce(&mut [Complex<Self>]) -> R>(len: usize, f: F) -> R;
  fn normal_array(n: usize, mean: Self, std_dev: Self) -> Array1<Self>;
}
