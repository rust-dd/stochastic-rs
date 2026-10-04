//! Truncated beta and gamma: rejection from the base law, with the clamped midpoint after 1000 rejections.
//!
//! Reference: Coffey, C. S. & Muller, K. E. (2000), "Properties of doubly-truncated gamma variables", *Communications in Statistics - Theory and Methods* 29(4), 851-857, DOI 10.1080/03610920008832519 (truncated gamma moments).
//! Reference: Johnson, N. L., Kotz, S. & Balakrishnan, N. (1995), *Continuous Univariate Distributions*, vol. 2, 2nd ed., Wiley, ch. 25, ISBN 0-471-58494-0 (beta moments; the truncated ones renormalise the incomplete beta ratio).

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use crate::beta::BetaState;
use crate::beta::SimdBeta;
use crate::gamma::GammaState;
use crate::gamma::SimdGamma;
use crate::traits::DistributionExt;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// The first base draw inside `[lower, upper]`, or the interval's midpoint once 1000 draws missed it.
#[inline]
fn reject<T: SimdFloatExt>(lower: T, upper: T, mut base: impl FnMut() -> T) -> T {
  for _ in 0..1000 {
    let x = base();
    if x >= lower && x <= upper {
      return x;
    }
  }
  (lower + upper) * T::from_f64_fast(0.5)
}

/// The parent's quantile at the renormalised level `F(lower) + p(F(upper) − F(lower))`.
fn renormalised_quantile(
  base: &impl DistributionExt,
  lower: f64,
  upper: f64,
  p: f64,
) -> Option<f64> {
  if !(0.0..=1.0).contains(&p) {
    return Some(f64::NAN);
  }
  let (f_lo, f_up) = (base.cdf(lower)?, base.cdf(upper)?);
  base.quantile(f_lo + p * (f_up - f_lo))
}

fn raw_variance(raw_moment: impl Fn(f64) -> Option<f64>) -> Option<f64> {
  let m1 = raw_moment(1.0)?;
  Some(raw_moment(2.0)? - m1 * m1)
}

/// Truncated beta law on $[\text{lower}, \text{upper}] \subseteq [0, 1]$, parameters only: a [`Seeded`](crate::Seeded)
/// stream draws it by rejection, falling back to the interval's midpoint after 1000 misses.
///
/// Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, §II.3, DOI 10.1007/978-1-4613-8643-8.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdTruncatedBeta<T> {
  base: SimdBeta<T>,
  lower: T,
  upper: T,
}

impl<T: SimdFloatExt> SimdTruncatedBeta<T> {
  /// The base [`SimdBeta`] `Beta(alpha, beta)`, both > 0, renormalised on `[lower, upper] ⊆ [0, 1]`, `lower < upper`.
  pub fn new(alpha: T, beta: T, lower: T, upper: T) -> Self {
    assert!(
      alpha.is_finite(),
      "alpha must satisfy `alpha.is_finite()`, got alpha = {alpha:?}"
    );
    assert!(
      beta.is_finite(),
      "beta must satisfy `beta.is_finite()`, got beta = {beta:?}"
    );
    assert!(
      !lower.is_nan(),
      "lower must satisfy `!lower.is_nan()`, got lower = {lower:?}"
    );
    assert!(
      !upper.is_nan(),
      "upper must satisfy `!upper.is_nan()`, got upper = {upper:?}"
    );
    assert!(
      alpha > T::zero(),
      "alpha must satisfy `alpha > T::zero()`, got alpha = {alpha:?}"
    );
    assert!(
      beta > T::zero(),
      "beta must satisfy `beta > T::zero()`, got beta = {beta:?}"
    );
    let lo = lower.to_f64().unwrap();
    let up = upper.to_f64().unwrap();
    assert!(
      (0.0..=1.0).contains(&lo),
      "lower must satisfy `0.0 <= lower <= 1.0`, got lower = {lower:?}"
    );
    assert!(
      (0.0..=1.0).contains(&up),
      "upper must satisfy `0.0 <= upper <= 1.0`, got upper = {upper:?}"
    );
    assert!(
      lower < upper,
      "lower must satisfy `lower < upper`, got lower = {lower:?}, upper = {upper:?}"
    );
    Self {
      base: SimdBeta::new(alpha, beta),
      lower,
      upper,
    }
  }

  /// The first shape `α` of the untruncated base.
  pub fn alpha(&self) -> T {
    self.base.alpha()
  }

  /// The second shape `β` of the untruncated base.
  pub fn beta(&self) -> T {
    self.base.beta()
  }

  /// The lower bound.
  pub fn lower(&self) -> T {
    self.lower
  }

  /// The upper bound.
  pub fn upper(&self) -> T {
    self.upper
  }

  fn cdf_helper(&self, x: f64) -> f64 {
    let a = self.base.alpha().to_f64().unwrap();
    let b = self.base.beta().to_f64().unwrap();
    crate::special::beta_i(a, b, x.clamp(0.0, 1.0))
  }

  /// $E[X^m]$ on the truncation: $B(\alpha+m, \beta)/B(\alpha, \beta)\,[I_b(\alpha+m, \beta) - I_a(\alpha+m, \beta)]/M$.
  fn raw_moment(&self, m: f64) -> Option<f64> {
    let (a, b) = (self.base.alpha().to_f64()?, self.base.beta().to_f64()?);
    let (lo, up) = (self.lower.to_f64()?, self.upper.to_f64()?);
    let mass = self.cdf_helper(up) - self.cdf_helper(lo);
    let shifted = crate::special::beta_i(a + m, b, up) - crate::special::beta_i(a + m, b, lo);
    let ratio = (crate::special::ln_beta(a + m, b) - crate::special::ln_beta(a, b)).exp();
    Some(ratio * shifted / mass)
  }
}

impl<T: SimdFloatExt> Sealed for SimdTruncatedBeta<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdTruncatedBeta<T> {
  type State<R: SimdRngExt> = BetaState<T, R>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (BetaState<T, R>, u64) {
    self.base.init::<R, S>(seed)
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdTruncatedBeta<T> {
  type Item = T;

  fn fill<R: SimdRngExt>(&self, state: &mut BetaState<T, R>, out: &mut [T]) {
    for x in out.iter_mut() {
      *x = self.next(state);
    }
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut BetaState<T, R>) -> T {
    reject(self.lower, self.upper, || self.base.next(state))
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdTruncatedBeta<T> {
  /// The same rejection on scalar beta draws from the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    reject(self.lower, self.upper, || self.base.draw_with(rng))
  }
}

impl<T: SimdFloatExt> DistributionExt for SimdTruncatedBeta<T> {
  fn pdf(&self, x: f64) -> Option<f64> {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x < lo || x > up {
      return Some(0.0);
    }
    let a = self.base.alpha().to_f64().unwrap();
    let b = self.base.beta().to_f64().unwrap();
    let log_norm =
      crate::special::ln_gamma(a + b) - crate::special::ln_gamma(a) - crate::special::ln_gamma(b);
    let log_kernel = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln();
    let base_pdf = (log_norm + log_kernel).exp();
    let norm_mass = self.cdf_helper(up) - self.cdf_helper(lo);
    Some(base_pdf / norm_mass.max(1e-300))
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x <= lo {
      return Some(0.0);
    }
    if x >= up {
      return Some(1.0);
    }
    let f_x = self.cdf_helper(x);
    let f_lo = self.cdf_helper(lo);
    let f_up = self.cdf_helper(up);
    Some((f_x - f_lo) / (f_up - f_lo).max(1e-300))
  }

  fn quantile(&self, p: f64) -> Option<f64> {
    renormalised_quantile(&self.base, self.lower.to_f64()?, self.upper.to_f64()?, p)
  }

  fn mean(&self) -> Option<f64> {
    self.raw_moment(1.0)
  }

  fn median(&self) -> Option<f64> {
    self.quantile(0.5)
  }

  /// Off the interior mode the density is monotone, U-shaped or flat on the interval: the higher end wins, a tie has no unique mode.
  fn mode(&self) -> Option<f64> {
    let (a, b) = (self.base.alpha().to_f64()?, self.base.beta().to_f64()?);
    let (lo, up) = (self.lower.to_f64()?, self.upper.to_f64()?);
    if a > 1.0 && b > 1.0 {
      return Some(((a - 1.0) / (a + b - 2.0)).clamp(lo, up));
    }
    let term = |e: f64, y: f64| if e == 0.0 { 0.0 } else { e * y.ln() };
    let log_kernel = |x: f64| term(a - 1.0, x) + term(b - 1.0, 1.0 - x);
    let (at_lo, at_up) = (log_kernel(lo), log_kernel(up));
    Some(if at_lo > at_up {
      lo
    } else if at_up > at_lo {
      up
    } else {
      f64::NAN
    })
  }

  fn variance(&self) -> Option<f64> {
    raw_variance(|m| self.raw_moment(m))
  }
}

/// Truncated $\mathrm{Gamma}(k, \theta)$ on $[\text{lower}, \text{upper}]$, $\text{lower} \ge 0$, parameters only: a
/// [`Seeded`](crate::Seeded) stream draws it by rejection, falling back to the interval's midpoint after 1000 misses.
///
/// Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, §II.3, DOI 10.1007/978-1-4613-8643-8.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdTruncatedGamma<T> {
  base: SimdGamma<T>,
  lower: T,
  upper: T,
}

impl<T: SimdFloatExt> SimdTruncatedGamma<T> {
  /// The base [`SimdGamma`] `Gamma(shape, scale)` (`shape` is its `alpha`), both > 0, renormalised on `[lower, upper]`,
  /// `0 ≤ lower < upper`.
  pub fn new(shape: T, scale: T, lower: T, upper: T) -> Self {
    assert!(
      shape.is_finite(),
      "shape must satisfy `shape.is_finite()`, got shape = {shape:?}"
    );
    assert!(
      scale.is_finite(),
      "scale must satisfy `scale.is_finite()`, got scale = {scale:?}"
    );
    assert!(
      !lower.is_nan(),
      "lower must satisfy `!lower.is_nan()`, got lower = {lower:?}"
    );
    assert!(
      !upper.is_nan(),
      "upper must satisfy `!upper.is_nan()`, got upper = {upper:?}"
    );
    assert!(
      shape > T::zero(),
      "shape must satisfy `shape > T::zero()`, got shape = {shape:?}"
    );
    assert!(
      scale > T::zero(),
      "scale must satisfy `scale > T::zero()`, got scale = {scale:?}"
    );
    assert!(
      lower >= T::zero(),
      "lower must satisfy `lower >= T::zero()`, got lower = {lower:?}"
    );
    assert!(
      lower < upper,
      "lower must satisfy `lower < upper`, got lower = {lower:?}, upper = {upper:?}"
    );
    Self {
      base: SimdGamma::new(shape, scale),
      lower,
      upper,
    }
  }

  /// The shape `k` of the untruncated base.
  pub fn shape(&self) -> T {
    self.base.alpha()
  }

  /// The scale `θ` of the untruncated base.
  pub fn scale(&self) -> T {
    self.base.scale()
  }

  /// The lower bound.
  pub fn lower(&self) -> T {
    self.lower
  }

  /// The upper bound.
  pub fn upper(&self) -> T {
    self.upper
  }

  fn cdf_helper(&self, x: f64) -> f64 {
    let k = self.base.alpha().to_f64().unwrap();
    let theta = self.base.scale().to_f64().unwrap();
    crate::special::gamma_p(k, x / theta)
  }

  /// $E[X^m]$ on the truncation: $\theta^m \Gamma(k+m)/\Gamma(k)\,[P(k+m, b/\theta) - P(k+m, a/\theta)]/M$.
  fn raw_moment(&self, m: f64) -> Option<f64> {
    let (k, theta) = (self.base.alpha().to_f64()?, self.base.scale().to_f64()?);
    let (lo, up) = (self.lower.to_f64()?, self.upper.to_f64()?);
    let mass = self.cdf_helper(up) - self.cdf_helper(lo);
    let shifted =
      crate::special::gamma_p(k + m, up / theta) - crate::special::gamma_p(k + m, lo / theta);
    let ratio = (crate::special::ln_gamma(k + m) - crate::special::ln_gamma(k)).exp();
    Some(theta.powf(m) * ratio * shifted / mass)
  }
}

impl<T: SimdFloatExt> Sealed for SimdTruncatedGamma<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdTruncatedGamma<T> {
  type State<R: SimdRngExt> = GammaState<T, R>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (GammaState<T, R>, u64) {
    self.base.init::<R, S>(seed)
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdTruncatedGamma<T> {
  type Item = T;

  fn fill<R: SimdRngExt>(&self, state: &mut GammaState<T, R>, out: &mut [T]) {
    for x in out.iter_mut() {
      *x = self.next(state);
    }
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut GammaState<T, R>) -> T {
    reject(self.lower, self.upper, || self.base.next(state))
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdTruncatedGamma<T> {
  /// The same rejection on scalar gamma draws from the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    reject(self.lower, self.upper, || self.base.draw_with(rng))
  }
}

impl<T: SimdFloatExt> DistributionExt for SimdTruncatedGamma<T> {
  fn pdf(&self, x: f64) -> Option<f64> {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x < lo || x > up {
      return Some(0.0);
    }
    let k = self.base.alpha().to_f64().unwrap();
    let theta = self.base.scale().to_f64().unwrap();
    let log_norm = -crate::special::ln_gamma(k) - k * theta.ln();
    let log_kernel = (k - 1.0) * x.ln() - x / theta;
    let base_pdf = (log_norm + log_kernel).exp();
    let mass = self.cdf_helper(up) - self.cdf_helper(lo);
    Some(base_pdf / mass.max(1e-300))
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x <= lo {
      return Some(0.0);
    }
    if x >= up {
      return Some(1.0);
    }
    let f_x = self.cdf_helper(x);
    let f_lo = self.cdf_helper(lo);
    let f_up = self.cdf_helper(up);
    Some((f_x - f_lo) / (f_up - f_lo).max(1e-300))
  }

  fn quantile(&self, p: f64) -> Option<f64> {
    renormalised_quantile(&self.base, self.lower.to_f64()?, self.upper.to_f64()?, p)
  }

  fn mean(&self) -> Option<f64> {
    self.raw_moment(1.0)
  }

  fn median(&self) -> Option<f64> {
    self.quantile(0.5)
  }

  fn mode(&self) -> Option<f64> {
    let (k, theta) = (self.base.alpha().to_f64()?, self.base.scale().to_f64()?);
    let (lo, up) = (self.lower.to_f64()?, self.upper.to_f64()?);
    Some(if k >= 1.0 {
      ((k - 1.0) * theta).clamp(lo, up)
    } else {
      lo
    })
  }

  fn variance(&self) -> Option<f64> {
    raw_variance(|m| self.raw_moment(m))
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::*;
  use crate::tests::scalar_ks_best_p;

  /// Truncated Beta in [0.2, 0.8] respects bounds and has uniform-like
  /// support across many samples.
  #[test]
  fn truncated_beta_samples_within_bounds() {
    let mut tb = SimdTruncatedBeta::<f64>::new(2.0, 2.0, 0.2, 0.8).seeded(&Unseeded);
    for _ in 0..3_000 {
      let x = tb.sample();
      assert!((0.2..=0.8).contains(&x));
    }
  }

  /// A bad `beta` shape is reported as `beta`, not as `alpha`.
  #[test]
  #[should_panic(expected = "beta must satisfy `beta > T::zero()`, got beta = -1.0")]
  fn truncated_beta_names_a_bad_beta() {
    SimdTruncatedBeta::<f64>::new(2.0, -1.0, 0.2, 0.8);
  }

  /// Truncated Gamma stays in bounds.
  #[test]
  fn truncated_gamma_samples_within_bounds() {
    let mut tg = SimdTruncatedGamma::<f64>::new(2.0, 1.0, 1.0, 5.0).seeded(&Unseeded);
    for _ in 0..3_000 {
      let x = tg.sample();
      assert!((1.0..=5.0).contains(&x));
    }
  }

  /// The honest `Distribution` of both laws draws from the caller's rng and agrees with the cdf.
  #[test]
  fn scalar_sample_matches_cdf() {
    let beta = SimdTruncatedBeta::<f64>::new(2.0, 2.0, 0.2, 0.8);
    let best = scalar_ks_best_p(&beta, |x| beta.cdf(x).unwrap());
    assert!(best > 0.01, "beta: best p = {best}");
    let gamma = SimdTruncatedGamma::<f64>::new(2.0, 1.0, 1.0, 5.0);
    let best = scalar_ks_best_p(&gamma, |x| gamma.cdf(x).unwrap());
    assert!(best > 0.01, "gamma: best p = {best}");
  }
}
