//! Foundational `RealExt`, `SimdFloatExt` and `FloatExt` traits.

use std::fmt::Debug;
use std::iter::Sum;
use std::mem::MaybeUninit;
use std::ops::AddAssign;
use std::ops::SubAssign;

use ndarray::Array1;
use ndarray::ScalarOperand;
use num_complex::Complex;
use rand::Rng;
use stochastic_rs_core::simd_rng::SimdRngExt;

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

  /// The larger of the two, NaN when either is NaN — `Float::max` returns the other operand instead.
  #[inline]
  fn max_or_nan(self, other: Self) -> Self {
    if self.is_nan() || other.is_nan() {
      Self::nan()
    } else {
      self.max(other)
    }
  }

  /// The smaller of the two, NaN when either is NaN.
  #[inline]
  fn min_or_nan(self, other: Self) -> Self {
    if self.is_nan() || other.is_nan() {
      Self::nan()
    } else {
      self.min(other)
    }
  }

  /// The largest of `items`, NaN if any is, `−∞` for none; the NaN flag is its own reduction, so
  /// the max still vectorises where a `max_or_nan` fold does not.
  #[inline]
  fn max_or_nan_fold<I: IntoIterator<Item = Self>>(items: I) -> Self {
    let (max, nan) = items
      .into_iter()
      .fold((Self::neg_infinity(), false), |(m, n), x| {
        (m.max(x), n | x.is_nan())
      });
    if nan { Self::nan() } else { max }
  }

  /// The smallest of `items`, NaN if any is, `+∞` for none.
  #[inline]
  fn min_or_nan_fold<I: IntoIterator<Item = Self>>(items: I) -> Self {
    let (min, nan) = items
      .into_iter()
      .fold((Self::infinity(), false), |(m, n), x| {
        (m.min(x), n | x.is_nan())
      });
    if nan { Self::nan() } else { min }
  }
}

mod sealed {
  #[diagnostic::on_unimplemented(
    message = "`{Self}` cannot implement the sealed trait `SimdFloatExt`",
    note = "`f32` and `f64` are the simulation scalars; a custom scalar stops at `RealExt`"
  )]
  pub trait Sealed {}

  impl Sealed for f32 {}

  impl Sealed for f64 {}
}

/// 8-lane SIMD surface over a [`RealExt`] scalar plus the uniform RNG fills the SIMD samplers
/// draw from; sealed, `f32` and `f64` are its only implementors.
pub trait SimdFloatExt: RealExt + sealed::Sealed {
  #[doc(hidden)]
  type Simd: Copy
    + std::ops::Mul<Output = Self::Simd>
    + std::ops::Add<Output = Self::Simd>
    + std::ops::Sub<Output = Self::Simd>
    + std::ops::Div<Output = Self::Simd>
    + std::ops::Neg<Output = Self::Simd>;

  #[doc(hidden)]
  fn splat(val: Self) -> Self::Simd;

  #[doc(hidden)]
  fn simd_from_array(arr: [Self; 8]) -> Self::Simd;

  #[doc(hidden)]
  fn simd_to_array(v: Self::Simd) -> [Self; 8];

  #[doc(hidden)]
  fn simd_ln(v: Self::Simd) -> Self::Simd;

  #[doc(hidden)]
  fn simd_sqrt(v: Self::Simd) -> Self::Simd;

  #[doc(hidden)]
  fn simd_cos(v: Self::Simd) -> Self::Simd;

  #[doc(hidden)]
  fn simd_sin(v: Self::Simd) -> Self::Simd;

  #[doc(hidden)]
  fn simd_exp(v: Self::Simd) -> Self::Simd;

  /// Lane-wise `e^v − 1`, exact where the subtraction alone would cancel.
  #[doc(hidden)]
  fn simd_exp_m1(v: Self::Simd) -> Self::Simd;

  #[doc(hidden)]
  fn simd_tan(v: Self::Simd) -> Self::Simd;

  #[doc(hidden)]
  fn simd_max(a: Self::Simd, b: Self::Simd) -> Self::Simd;

  #[doc(hidden)]
  fn simd_powf(v: Self::Simd, exp: Self) -> Self::Simd;

  fn fill_uniform<R: Rng + ?Sized>(rng: &mut R, out: &mut [Self]);

  fn fill_uniform_simd<R: SimdRngExt>(rng: &mut R, out: &mut [Self]);

  /// `fill_uniform_simd` into uninitialised storage; writes every element and reads none, so a
  /// caller may `assume_init`.
  #[doc(hidden)]
  fn fill_uniform_simd_uninit<R: SimdRngExt>(rng: &mut R, out: &mut [MaybeUninit<Self>]);

  fn sample_uniform<R: Rng + ?Sized>(rng: &mut R) -> Self;

  #[inline(always)]
  fn sample_uniform_simd<R: SimdRngExt>(rng: &mut R) -> Self {
    let mut buf = [Self::zero(); 8];
    Self::fill_uniform_simd(rng, &mut buf);
    buf[0]
  }

  #[doc(hidden)]
  fn simd_from_i32x8(v: wide::i32x8) -> Self::Simd;

  #[doc(hidden)]
  const PREFERS_F32_WN: bool = false;

  /// Markov-lift factor reduction $\sum_l (w_l e_l)(H_l + J_l)$ for one path; `we` is `w_l e_l`
  /// pre-merged; panics unless the slices share one length.
  #[doc(hidden)]
  fn history_sum_fused(we: &[Self], h_state: &[Self], j_state: &[Self]) -> Self;

  /// One path's factor-state update, `omx` holding $(1 - e_l)/x_l$; panics unless the slices share one length.
  #[doc(hidden)]
  fn update_state_fused(
    h_state: &mut [Self],
    j_state: &mut [Self],
    exp_neg: &[Self],
    omx: &[Self],
    f_prev: Self,
    g_dw: Self,
  );

  /// Adds `we_l (h_row + j_row)` to every path's `history`; panics unless the slices share one length.
  #[doc(hidden)]
  fn batch_history_accumulate(we_l: Self, h_row: &[Self], j_row: &[Self], history: &mut [Self]);

  /// Factor $l$'s state update across every path; panics unless the slices share one length.
  #[doc(hidden)]
  fn batch_update_state(
    e_l: Self,
    omx_l: Self,
    h_row: &mut [Self],
    j_row: &mut [Self],
    f_prev: &[Self],
    g_dw: &[Self],
  );
}

/// The full simulation-grade float: [`RealExt`] arithmetic, [`SimdFloatExt`] lanes and the batched
/// normal / fGN scratch every sampler draws through; sealed through `SimdFloatExt`.
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
  fn normal_array(n: usize, mu: Self, sigma: Self) -> Array1<Self>;
}

#[cfg(test)]
mod tests {
  use super::RealExt;

  #[test]
  fn max_and_min_or_nan_folds() {
    assert_eq!(f64::max_or_nan_fold([1.0, 3.0, 2.0]), 3.0);
    assert_eq!(f64::min_or_nan_fold([1.0, 3.0, 2.0]), 1.0);
    assert_eq!(f64::max_or_nan_fold(std::iter::empty()), f64::NEG_INFINITY);
    assert_eq!(f64::min_or_nan_fold(std::iter::empty()), f64::INFINITY);
    assert!(f64::max_or_nan_fold([1.0, f64::NAN, 2.0]).is_nan());
    assert!(f32::min_or_nan_fold([1.0_f32, f32::NAN]).is_nan());
    assert_eq!(f64::max_or_nan_fold([f64::NEG_INFINITY, -1.0]), -1.0);
    assert_eq!(f64::min_or_nan_fold([f64::INFINITY, 1.0]), 1.0);
    let xs = (0..1_000).map(|i| (i as f64).sin()).collect::<Vec<_>>();
    assert_eq!(
      f64::max_or_nan_fold(xs.iter().copied()),
      xs.iter().copied().fold(f64::NEG_INFINITY, f64::max_or_nan)
    );
  }

  #[test]
  fn a_nan_wins_at_every_position_of_every_short_slice() {
    for n in 1..=33_usize {
      let xs = (0..n)
        .map(|i| ((i * 7) % 11) as f64 - 5.0)
        .collect::<Vec<_>>();
      assert_eq!(
        f64::max_or_nan_fold(xs.iter().copied()),
        xs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
      );
      assert_eq!(
        f64::min_or_nan_fold(xs.iter().copied()),
        xs.iter().copied().fold(f64::INFINITY, f64::min)
      );
      for at in 0..n {
        let mut with_nan = xs.clone();
        with_nan[at] = f64::NAN;
        assert!(f64::max_or_nan_fold(with_nan.iter().copied()).is_nan());
        assert!(f64::min_or_nan_fold(with_nan.iter().copied()).is_nan());
      }
    }
  }
}
