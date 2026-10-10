//! # Pareto
//!
//! $$
//! f(x)=\alpha x_m^\alpha x^{-(\alpha+1)},\ x\ge x_m
//! $$
//!
//! Sampling: inversion, Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, §II.2, DOI 10.1007/978-1-4613-8643-8.

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use super::SimdFloatExt;
use crate::seeded::StreamState;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

const SMALL_PARETO_THRESHOLD: usize = 16;

/// Pareto (Type I) law with scale `x_m` and tail index `alpha`: parameters only; a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdPareto<T> {
  x_m: T,
  alpha: T,
}

impl<T: SimdFloatExt> SimdPareto<T> {
  /// Creates a Pareto (Type I) distribution.
  ///
  /// - `x_m` — minimum/scale x_m > 0 (matches the module header's x_m);
  ///   also the mode.
  /// - `alpha` — tail index α > 0 (matches the module header's α);
  ///   controls which moments exist (mean requires α>1, variance α>2).
  pub fn new(x_m: T, alpha: T) -> Self {
    assert!(
      x_m.is_finite(),
      "x_m must satisfy `x_m.is_finite()`, got x_m = {x_m:?}"
    );
    assert!(
      alpha.is_finite(),
      "alpha must satisfy `alpha.is_finite()`, got alpha = {alpha:?}"
    );
    assert!(
      x_m > T::zero(),
      "x_m must satisfy `x_m > T::zero()`, got x_m = {x_m:?}"
    );
    assert!(
      alpha > T::zero(),
      "alpha must satisfy `alpha > T::zero()`, got alpha = {alpha:?}"
    );
    Self { x_m, alpha }
  }

  /// The scale `x_m`.
  pub fn x_m(&self) -> T {
    self.x_m
  }

  /// The tail index `α`.
  pub fn alpha(&self) -> T {
    self.alpha
  }

  /// `x_m·(1 − u)^{−1/α}`, the inverse cdf at `u`; `1 − u` is held at the smallest positive value.
  #[inline]
  fn invert(&self, u: T, neg_inv_alpha: T) -> T {
    let base = (T::one() - u).max(T::min_positive_val());
    self.x_m * (base.ln() * neg_inv_alpha).exp()
  }

  fn fill_parts<R: SimdRngExt>(&self, rng: &mut R, out: &mut [T]) {
    if out.len() < SMALL_PARETO_THRESHOLD {
      let neg_inv_alpha = -T::one() / self.alpha;
      for x in out.iter_mut() {
        *x = self.invert(T::sample_uniform_simd(rng), neg_inv_alpha);
      }
      return;
    }
    let xm = T::splat(self.x_m);
    let neg_inv_alpha = T::splat(-T::one() / self.alpha);
    let one = T::splat(T::one());
    let eps = T::splat(T::min_positive_val());
    let mut u = [T::zero(); 8];
    let (chunks, rem) = out.as_chunks_mut::<8>();
    for chunk in chunks {
      T::fill_uniform_simd(rng, &mut u);
      let v = T::simd_from_array(u);
      let base = T::simd_max(one - v, eps);
      let x = xm * T::simd_exp(T::simd_ln(base) * neg_inv_alpha);
      *chunk = T::simd_to_array(x);
    }
    if !rem.is_empty() {
      T::fill_uniform_simd(rng, &mut u);
      let v = T::simd_from_array(u);
      let base = T::simd_max(one - v, eps);
      let x = T::simd_to_array(xm * T::simd_exp(T::simd_ln(base) * neg_inv_alpha));
      rem.copy_from_slice(&x[..rem.len()]);
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.invert(T::sample_uniform(rng), -T::one() / self.alpha)
  }
}

impl<T: SimdFloatExt> Sealed for SimdPareto<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdPareto<T> {
  type State<R: SimdRngExt> = StreamState<T, R, 16>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (StreamState<T, R, 16>, u64) {
    StreamState::init(seed)
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdPareto<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 16>, out: &mut [T]) {
    self.fill_parts(&mut state.rng, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 16>) -> T {
    let StreamState { rng, buf } = state;
    buf.pop(|b| self.fill_parts(rng, b))
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdPareto<T> {
  /// The inverse cdf at one `[0, 1)` uniform from the caller's rng (53 bits for `f64`, 24 for `f32`).
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdPareto<T> {
  fn pdf(&self, x: f64) -> Option<f64> {
    let xm = self.x_m.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    if x < xm {
      Some(0.0)
    } else {
      Some(a * xm.powf(a) / x.powf(a + 1.0))
    }
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    let xm = self.x_m.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    if x < xm {
      Some(0.0)
    } else {
      Some(1.0 - (xm / x).powf(a))
    }
  }

  fn quantile(&self, p: f64) -> Option<f64> {
    let xm = self.x_m.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    Some(xm / (1.0 - p).powf(1.0 / a))
  }

  /// `+∞`, not `NaN`, at `alpha <= 1`: the mean integral `∫x·f(x)dx`
  /// diverges to a definite (infinite) value at these — commonly used —
  /// shape parameters (e.g. the classic "80/20" Pareto has `alpha ≈ 1.16`).
  fn mean(&self) -> Option<f64> {
    let xm = self.x_m.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    if a > 1.0 {
      Some(xm * a / (a - 1.0))
    } else {
      Some(f64::INFINITY)
    }
  }

  fn median(&self) -> Option<f64> {
    let xm = self.x_m.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    Some(xm * 2.0_f64.powf(1.0 / a))
  }

  fn mode(&self) -> Option<f64> {
    Some(self.x_m.to_f64().unwrap())
  }

  /// `+∞`, not `NaN`, at `alpha <= 2`: same divergent-integral reason as
  /// `mean`, one moment order up.
  fn variance(&self) -> Option<f64> {
    let xm = self.x_m.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    if a > 2.0 {
      Some(xm * xm * a / ((a - 1.0).powi(2) * (a - 2.0)))
    } else {
      Some(f64::INFINITY)
    }
  }

  /// `NaN`, not `+∞`, at `alpha <= 3`: unlike mean/variance, the third
  /// central moment does not merely diverge to a signed infinity here — it
  /// is not defined at all, so `NaN` is the honest answer rather than a
  /// sign choice.
  fn skewness(&self) -> Option<f64> {
    let a = self.alpha.to_f64().unwrap();
    if a > 3.0 {
      Some(2.0 * (1.0 + a) / (a - 3.0) * ((a - 2.0) / a).sqrt())
    } else {
      Some(f64::NAN)
    }
  }

  /// `NaN` at `alpha <= 4`, for the same reason as `skewness` one moment
  /// order up.
  fn kurtosis(&self) -> Option<f64> {
    let a = self.alpha.to_f64().unwrap();
    if a > 4.0 {
      Some(6.0 * (a.powi(3) + a.powi(2) - 6.0 * a - 2.0) / (a * (a - 3.0) * (a - 4.0)))
    } else {
      Some(f64::NAN)
    }
  }

  fn entropy(&self) -> Option<f64> {
    let xm = self.x_m.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    Some((xm / a).ln() + 1.0 / a + 1.0)
  }

  /// `NaN` for `t > 0`, where the polynomial tail diverges; `None` for `t < 0`, where the mgf is
  /// `α(−x_m t)^α Γ(−α, −x_m t)` and `special` has no incomplete gamma of negative order.
  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    if t < 0.0 {
      None
    } else if t == 0.0 {
      Some(1.0)
    } else {
      Some(f64::NAN)
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::tests::scalar_ks_best_p;
  use crate::traits::DistributionExt;

  /// Backs the doc comments on `mean`/`variance`/`skewness`/`kurtosis`:
  /// `alpha = 1.16` (the classic "80/20" Pareto) is a common, valid,
  /// in-range shape parameter, yet already sits below every one of these
  /// thresholds — mean is the only finite moment it has.
  #[test]
  fn pareto_80_20_moments_match_documented_thresholds() {
    let p = SimdPareto::<f64>::new(1.0, 1.16);
    assert!(
      p.mean().unwrap().is_finite(),
      "alpha=1.16 > 1, mean should be finite"
    );
    assert_eq!(p.variance().unwrap(), f64::INFINITY, "alpha=1.16 <= 2");
    assert!(p.skewness().unwrap().is_nan(), "alpha=1.16 <= 3");
    assert!(p.kurtosis().unwrap().is_nan(), "alpha=1.16 <= 4");
    assert!(
      p.moment_generating_function(0.5).unwrap().is_nan(),
      "MGF at t > 0 must be NaN"
    );
  }

  /// No mgf for t > 0, 1 at t = 0, and for t < 0 one that exists but has no closed form here.
  #[test]
  fn mgf_answers_by_the_sign_of_t() {
    let p = SimdPareto::<f64>::new(1.0, 3.0);
    assert!(p.moment_generating_function(0.4).unwrap().is_nan());
    assert_eq!(p.moment_generating_function(0.0), Some(1.0));
    assert!(p.moment_generating_function(-0.4).is_none());
  }

  /// Above every threshold, all four moments must be finite real numbers.
  #[test]
  fn pareto_high_alpha_moments_are_all_finite() {
    let p = SimdPareto::<f64>::new(1.0, 5.0);
    assert!(p.mean().unwrap().is_finite());
    assert!(p.variance().unwrap().is_finite());
    assert!(p.skewness().unwrap().is_finite());
    assert!(p.kurtosis().unwrap().is_finite());
  }

  /// The honest `Distribution` draws from the caller's rng and agrees with the cdf.
  #[test]
  fn scalar_sample_matches_cdf() {
    let d = SimdPareto::<f64>::new(1.0, 1.16);
    let best = scalar_ks_best_p(&d, |x| d.cdf(x).unwrap());
    assert!(best > 0.01, "best p = {best}");
  }
}

py_distribution!(PyPareto, SimdPareto,
  sig: (x_m, alpha, seed=None, dtype=None),
  params: (x_m: f64, alpha: f64)
);
