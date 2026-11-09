//! # Geometric
//!
//! $$
//! \mathbb{P}(X=k)=(1-p)^{k-1}p,\ k\ge 1
//! $$
//!
//! Sampling: inversion `⌊ln(1 − U)/ln(1 − p)⌋ + 1` with `ln(1 − p)` as `ln_1p(−p)`, Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, §X.2, DOI 10.1007/978-1-4613-8643-8.

use std::marker::PhantomData;
use std::mem::MaybeUninit;

use num_traits::PrimInt;
use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;
use stochastic_rs_core::simd_rng::as_uninit_mut;
use wide::f64x8;

use crate::seeded::StreamState;
use crate::source::uniform53;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

const SMALL_GEOMETRIC_THRESHOLD: usize = 16;

/// Geometric law on `{1, 2, …}` with success probability `p`: parameters only; a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdGeometric<T: PrimInt> {
  p: f64,
  /// `1/ln(1 − p)` through `ln_1p`, stored so a single draw takes no logarithm of `p`.
  inv_ln_q: f64,
  out: PhantomData<T>,
}

impl<T: PrimInt> SimdGeometric<T> {
  /// Trials up to and including the first success, support `k ≥ 1`; panics unless `p ∈ (0, 1]`.
  pub fn new(p: f64) -> Self {
    assert!(
      p.is_finite(),
      "p must satisfy `p.is_finite()`, got p = {p:?}"
    );
    assert!(
      p > 0.0 && p <= 1.0,
      "p must satisfy `p > 0.0 && p <= 1.0`, got p = {p:?}"
    );
    Self {
      p,
      inv_ln_q: 1.0 / (-p).ln_1p(),
      out: PhantomData,
    }
  }

  /// The success probability `p`.
  pub fn p(&self) -> f64 {
    self.p
  }

  /// The trial count of the floored inversion `t = ⌊ln(1 − u) / ln(1 − p)⌋`, shifted onto
  /// `{1, 2, …}`; an overflow of `T` saturates (and asserts in debug).
  #[inline]
  fn count(t: f64) -> T {
    let k = t.max(0.0) + 1.0;
    let cast = num_traits::cast(k as u64);
    debug_assert!(
      cast.is_some(),
      "geometric draw {k} overflowed the output integer type"
    );
    cast.unwrap_or(T::max_value())
  }

  fn fill_parts<R: SimdRngExt>(&self, rng: &mut R, out: &mut [MaybeUninit<T>]) {
    if out.len() < SMALL_GEOMETRIC_THRESHOLD {
      for x in out.iter_mut() {
        x.write(Self::count(
          ((1.0 - rng.next_f64()).ln() * self.inv_ln_q).floor(),
        ));
      }
      return;
    }
    let inv_ln1p = f64x8::splat(self.inv_ln_q);
    let (chunks, rem) = out.as_chunks_mut::<8>();
    for chunk in chunks {
      let mut u = [0.0_f64; 8];
      rng.fill_uniform_f64(&mut u);
      let tmp = ((f64x8::ONE - f64x8::from(u)).ln() * inv_ln1p)
        .floor()
        .to_array();
      for (o, &t) in chunk.iter_mut().zip(tmp.iter()) {
        o.write(Self::count(t));
      }
    }
    if !rem.is_empty() {
      let mut u = [0.0_f64; 8];
      rng.fill_uniform_f64(&mut u);
      let tmp = ((f64x8::ONE - f64x8::from(u)).ln() * inv_ln1p)
        .floor()
        .to_array();
      for (o, &t) in rem.iter_mut().zip(tmp.iter()) {
        o.write(Self::count(t));
      }
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    Self::count(((1.0 - uniform53(rng.next_u64())).ln() * self.inv_ln_q).floor())
  }
}

impl<T: PrimInt + Send + Sync + 'static> Sealed for SimdGeometric<T> {}

impl<T: PrimInt + Send + Sync + 'static> SimdDistribution for SimdGeometric<T> {
  type State<R: SimdRngExt> = StreamState<T, R, 16>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (StreamState<T, R, 16>, u64) {
    StreamState::init(seed)
  }
}

impl<T: PrimInt + Send + Sync + 'static> SimdKernel for SimdGeometric<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 16>, out: &mut [MaybeUninit<T>]) {
    self.fill_parts(&mut state.rng, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 16>) -> T {
    let StreamState { rng, buf } = state;
    // SAFETY: `fill_parts` writes every element of the buffer.
    buf.pop(|b| self.fill_parts(rng, unsafe { as_uninit_mut(b) }))
  }
}

impl<T: PrimInt + Send + Sync + 'static> Distribution<T> for SimdGeometric<T> {
  /// The inversion of one 53-bit uniform from the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: PrimInt> crate::traits::DistributionExt for SimdGeometric<T> {
  // Convention here: support k ∈ {1, 2, ...} (the "shifted" geometric, P(X=k) = (1-p)^(k-1) p).

  fn pdf(&self, x: f64) -> Option<f64> {
    if x.is_nan() {
      return Some(f64::NAN);
    }
    if x < 1.0 || x.fract() != 0.0 {
      return Some(0.0);
    }
    // At `p = 1` the general form below reads `0 · ln 0` for `x = 1`.
    if x == 1.0 {
      return Some(self.p);
    }
    Some(self.p * ((x - 1.0) * (-self.p).ln_1p()).exp())
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    if x.is_nan() {
      return Some(f64::NAN);
    }
    if x < 1.0 {
      return Some(0.0);
    }
    Some(-(x.floor() * (-self.p).ln_1p()).exp_m1())
  }

  fn quantile(&self, prob: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&prob) {
      return Some(f64::NAN);
    }
    // At `p = 1` the support is `{1}`; the formula below would read `ln 0` there.
    if prob == 0.0 || self.p == 1.0 {
      return Some(1.0);
    }
    if prob == 1.0 {
      return Some(f64::INFINITY);
    }
    // Smallest k such that 1-(1-p)^k ≥ prob ⟹ k = ⌈ln(1-prob)/ln(1-p)⌉
    Some(((-prob).ln_1p() / (-self.p).ln_1p()).ceil())
  }

  fn mean(&self) -> Option<f64> {
    Some(1.0 / self.p)
  }

  fn median(&self) -> Option<f64> {
    self.quantile(0.5)
  }

  fn mode(&self) -> Option<f64> {
    Some(1.0)
  }

  fn variance(&self) -> Option<f64> {
    Some((1.0 - self.p) / (self.p * self.p))
  }

  fn skewness(&self) -> Option<f64> {
    Some((2.0 - self.p) / (1.0 - self.p).sqrt())
  }

  fn kurtosis(&self) -> Option<f64> {
    Some(6.0 + self.p * self.p / (1.0 - self.p))
  }

  /// `0·ln 0 = 0` at `p = 1`, the law that always succeeds first, so its entropy is 0 rather than
  /// NaN.
  fn entropy(&self) -> Option<f64> {
    let q = 1.0 - self.p;
    let q_term = if q > 0.0 { q * q.ln() } else { 0.0 };
    Some(-(q_term + self.p * self.p.ln()) / self.p)
  }

  fn characteristic_function(&self, t: f64) -> Option<num_complex::Complex64> {
    if t == 0.0 {
      return Some(num_complex::Complex64::new(1.0, 0.0));
    }
    // φ(t) = p e^{it} / (1 - (1-p) e^{it})
    let eit = num_complex::Complex64::new(0.0, t).exp();
    Some(eit.scale(self.p) / (num_complex::Complex64::new(1.0, 0.0) - eit.scale(1.0 - self.p)))
  }

  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    if t.is_nan() {
      return Some(f64::NAN);
    }
    let q = 1.0 - self.p;
    if t == 0.0 {
      Some(1.0)
    } else if q * t.exp() < 1.0 {
      Some(self.p * t.exp() / (1.0 - q * t.exp()))
    } else {
      Some(f64::INFINITY)
    }
  }
}

py_distribution_int!(PyGeometric, SimdGeometric,
  sig: (p, seed=None),
  params: (p: f64)
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::tests::scalar_chi_square_best_p;
  use crate::traits::DistributionExt;
  use crate::traits::DistributionSampler;

  /// At `p = 1` the entropy is 0, not the NaN of a plain `0·ln 0`.
  #[test]
  fn entropy_at_p_one_is_zero_not_nan() {
    let g = SimdGeometric::<u64>::new(1.0);
    assert_eq!(
      g.entropy().unwrap(),
      0.0,
      "p=1.0 is degenerate, entropy must be exactly 0"
    );
  }

  /// Away from the boundary, entropy must still be a finite, positive
  /// number (uncertainty is strictly positive whenever `p < 1`).
  #[test]
  fn entropy_at_interior_p_is_finite_and_positive() {
    let g = SimdGeometric::<u64>::new(0.3);
    let h = g.entropy().unwrap();
    assert!(h.is_finite() && h > 0.0, "entropy at p=0.3 was {h}");
  }

  /// `p = 1`, the closed upper end of `(0, 1]`, constructs.
  #[test]
  fn new_accepts_p_one() {
    let _ = SimdGeometric::<u64>::new(1.0);
  }

  /// `p = 0`, outside `(0, 1]` (the wait never ends), panics at construction.
  #[test]
  #[should_panic(expected = "p must satisfy `p > 0.0 && p <= 1.0`")]
  fn new_rejects_p_zero() {
    let _ = SimdGeometric::<u64>::new(0.0);
  }

  /// `p > 1.0` is not a probability at all and must be rejected the same
  /// way `p = 0.0` is.
  #[test]
  #[should_panic(expected = "p must satisfy `p > 0.0 && p <= 1.0`")]
  fn new_rejects_p_above_one() {
    let _ = SimdGeometric::<u64>::new(1.5);
  }

  /// `fill_slice`'s mean and variance match `mean()` and `variance()` within their standard errors
  /// across `p`, `p = 1` included: the sampler and analytics share the `k ≥ 1` support.
  #[test]
  fn sampler_moments_match_analytics_across_p() {
    const N: usize = 200_000;
    for &p in &[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] {
      let dist = SimdGeometric::<u64>::new(p);
      let mut buf = vec![0u64; N];
      dist.seeded(&Deterministic::new(7)).fill_slice(&mut buf);

      if p >= 1.0 {
        let bad = buf.iter().filter(|&&x| x != 1).count();
        assert_eq!(bad, 0, "p=1.0: {bad}/{N} draws were not 1");
        assert_eq!(
          dist.mean().unwrap(),
          1.0,
          "p=1.0: mean() must be exactly 1.0"
        );
        assert_eq!(
          dist.variance().unwrap(),
          0.0,
          "p=1.0: variance() must be exactly 0.0"
        );
        continue;
      }

      let n = N as f64;
      let mean = buf.iter().map(|&x| x as f64).sum::<f64>() / n;
      let var = buf
        .iter()
        .map(|&x| {
          let d = x as f64 - mean;
          d * d
        })
        .sum::<f64>()
        / n;

      let expected_mean = dist.mean().unwrap();
      let expected_var = dist.variance().unwrap();
      // `μ₄` from the shift-invariant kurtosis gives the variance estimator's error,
      // `Var(S²) ≈ (μ₄ − σ⁴)/n`.
      let mu4 = (dist.kurtosis().unwrap() + 3.0) * expected_var * expected_var;
      let se_mean = (expected_var / n).sqrt();
      let se_var = ((mu4 - expected_var * expected_var) / n).sqrt();

      assert!(
        (mean - expected_mean).abs() < 6.0 * se_mean,
        "p={p}: sample mean {mean} vs mean() {expected_mean} (6*SE={})",
        6.0 * se_mean
      );
      assert!(
        (var - expected_var).abs() < 6.0 * se_var,
        "p={p}: sample variance {var} vs variance() {expected_var} (6*SE={})",
        6.0 * se_var
      );
    }
  }

  /// Buffers under `SMALL_GEOMETRIC_THRESHOLD` take the scalar path, with the SIMD path's `+ 1`
  /// shift.
  #[test]
  fn scalar_path_respects_shifted_support() {
    let mut dist = SimdGeometric::<u64>::new(0.4).seeded(&Deterministic::new(5));
    let mut buf = [0u64; 8];
    dist.fill_slice(&mut buf);
    assert!(buf.iter().all(|&x| x >= 1), "scalar-path draws: {buf:?}");
  }

  /// The honest `Distribution` draws from the caller's rng and agrees with the cdf.
  #[test]
  fn scalar_sample_matches_cdf() {
    let d = SimdGeometric::<u64>::new(0.15);
    let best = scalar_chi_square_best_p(&d, (1, 50), |k| d.cdf(k as f64).unwrap());
    assert!(best > 0.01, "best p = {best}");
  }

  /// NaN answers NaN, and counts past `i32::MAX` or a small `p`, where `1 − (1 − p)^k` cancels,
  /// keep the closed form; references from mpmath.
  #[test]
  fn cdf_and_pmf_follow_the_closed_form_at_any_count() {
    let g = SimdGeometric::<u64>::new(0.3);
    assert!(g.cdf(f64::NAN).unwrap().is_nan() && g.pdf(f64::NAN).unwrap().is_nan());
    for x in [-1.0, 0.0] {
      assert_eq!((g.cdf(x), g.pdf(x)), (Some(0.0), Some(0.0)), "x = {x}");
    }
    assert!((g.cdf(3.0).unwrap() - 0.657).abs() < 1e-15);
    assert!((g.pdf(3.0).unwrap() - 0.147).abs() < 1e-15);
    for x in [3e9, 1e10, f64::INFINITY] {
      assert_eq!((g.cdf(x), g.pdf(x)), (Some(1.0), Some(0.0)), "x = {x}");
    }
    let rare = SimdGeometric::<u64>::new(1e-10);
    let rel = |got: f64, want: f64| (got - want).abs() / want;
    assert!(rel(rare.cdf(3.0).unwrap(), 2.999_999_999_7e-10) < 1e-14);
    assert!(rel(rare.cdf(1e10).unwrap(), 0.632_120_558_846_951_6) < 1e-14);
    assert!(rel(rare.pdf(1e10).unwrap(), 3.678_794_411_898_363e-11) < 1e-14);
    let sure = SimdGeometric::<u64>::new(1.0);
    assert_eq!(
      (sure.pdf(1.0), sure.pdf(2.0), sure.cdf(1.0)),
      (Some(1.0), Some(0.0), Some(1.0))
    );
  }

  /// At `p = 1` every level's quantile is the one support point, as every draw is.
  #[test]
  fn quantile_at_p_one_is_the_support_point() {
    let sure = SimdGeometric::<u64>::new(1.0);
    for prob in [0.0, 1e-300, 0.5, 1.0 - f64::EPSILON, 1.0] {
      assert_eq!(sure.quantile(prob), Some(1.0), "prob = {prob}");
    }
  }

  /// `ln(1 − p)` through `ln_1p` and `ln(1 − prob)` likewise: at p = 1e-10 the plain logs lose eight digits (mpmath exact counts).
  #[test]
  fn quantile_and_median_keep_their_digits_at_small_p() {
    let rare = SimdGeometric::<u64>::new(1e-10);
    assert_eq!(rare.quantile(0.3), Some(3_566_749_440.0));
    assert_eq!(rare.quantile(0.5), Some(6_931_471_806.0));
    assert_eq!(rare.median(), Some(6_931_471_806.0));
    assert_eq!(SimdGeometric::<u64>::new(0.15).quantile(0.999), Some(43.0));
  }

  /// A rare success: the sampler's counts agree with the cdf over the bulk of the support.
  #[test]
  fn scalar_sample_matches_cdf_at_small_p() {
    let d = SimdGeometric::<u64>::new(1e-4);
    let best = scalar_chi_square_best_p(&d, (1, 50_000), |k| d.cdf(k as f64).unwrap());
    assert!(best > 0.01, "best p = {best}");
  }
}
