//! Foundational `FloatExt` and `SimdFloatExt` traits.

use std::fmt::Debug;
use std::iter::Sum;
use std::ops::AddAssign;
use std::ops::SubAssign;

use ndarray::Array1;
use ndarray::ScalarOperand;
use num_complex::Complex;
use rand::Rng;

pub trait SimdFloatExt: num_traits::Float + Default + Send + Sync + 'static {
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
  fn simd_floor(v: Self::Simd) -> Self::Simd;
  /// `ln(1 + v)` — accurate for small `|v|` where `ln(1 + v)` cancels.
  fn simd_log1p(v: Self::Simd) -> Self::Simd;
  /// `exp(v) - 1` — accurate for small `|v|` where `exp(v) - 1` cancels.
  fn simd_expm1(v: Self::Simd) -> Self::Simd;
  /// `sqrt(a² + b²)` with overflow protection.
  fn simd_hypot(a: Self::Simd, b: Self::Simd) -> Self::Simd;
  /// Fused multiply-add `a * b + c` — exposes the hardware FMA when
  /// available and a tight `mul_add` fallback otherwise.
  fn simd_fma(a: Self::Simd, b: Self::Simd, c: Self::Simd) -> Self::Simd;
  /// Error function $\mathrm{erf}(x)$. Accurate to ≈ 1.5e-7 in `f32`,
  /// ≈ 1.5e-12 in `f64` (Abramowitz-Stegun 7.1.26 rational approximation).
  fn simd_erf(v: Self::Simd) -> Self::Simd;
  /// Complementary error function $\mathrm{erfc}(x) = 1 - \mathrm{erf}(x)$.
  /// Uses the same A&S rational approximation; tails are stable down to
  /// `|x| < 6` (`f32`) / `|x| < 26` (`f64`).
  fn simd_erfc(v: Self::Simd) -> Self::Simd;
  /// Logarithm of the absolute Gamma function $\ln |\Gamma(x)|$.
  /// Stirling-based with Lanczos correction; mean relative error
  /// ≈ 1e-7 (`f32`) / ≈ 1e-12 (`f64`) for `x ∈ (0, 100]`.
  fn simd_lgamma(v: Self::Simd) -> Self::Simd;
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
  fn from_f64_fast(v: f64) -> Self;
  #[inline(always)]
  fn from_f32_fast(v: f32) -> Self {
    Self::from_f64_fast(v as f64)
  }
  fn pi() -> Self;
  fn two_pi() -> Self;
  fn min_positive_val() -> Self;
}

pub trait FloatExt:
  num_traits::Float
  + num_traits::FromPrimitive
  + num_traits::Signed
  + num_traits::FloatConst
  + Sum
  + SimdFloatExt
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
