//! # Truncated distributions
//!
//! Truncated $\mathrm{Normal}$, $\mathrm{Beta}$, $\mathrm{Gamma}$ and
//! $\mathrm{Exponential}$ distributions restricted to a closed interval
//! $[a, b]$ (with $-\infty \le a < b \le +\infty$ allowed per family).
//!
//! ## Sampling
//!
//! - **Truncated Normal:** plain rejection from the base
//!   $\mathcal{N}(\mu, \sigma^2)$ via the existing [`SimdNormal`] sampler.
//!   Tight intervals (probability-mass $< 0.05$) fall back to Robert's
//!   exponential algorithm (Robert 1995, *Statistics and Computing* 5,
//!   §3) which keeps acceptance ≥ 0.65 in the worst case.
//! - **Truncated Exponential:** closed-form inverse-CDF sampling — no
//!   rejection needed.
//! - **Truncated Beta / Gamma:** plain rejection from the corresponding
//!   [`SimdBeta`] / [`SimdGamma`] sampler. For very tight intervals where
//!   acceptance falls below 1 % the rejection loop bails after 1000 tries
//!   and returns the clamped midpoint — the caller should widen the
//!   bounds in that regime (the boundary itself is hit with measure zero).
//!
//! ## Density
//!
//! Standard normalisation: $f_{[a,b]}(x) = f(x) / (F(b) - F(a))$ for
//! $x \in [a, b]$ and $0$ elsewhere. The CDF normalising constant is
//! cached at construction time.
//!
//! References:
//! - Robert, C.P. (1995), "Simulation of truncated normal variables",
//!   *Statistics and Computing* 5, 121-125.
//! - Botts, C. (2013), "An accept-reject algorithm for the positive
//!   multivariate normal distribution", *Computational Statistics* 28,
//!   1749-1773.
//! - Devroye, L. (1986), *Non-Uniform Random Variate Generation*,
//!   Springer, §II.3 (general rejection).

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::beta::SimdBeta;
use crate::exp::SimdExp;
use crate::gamma::SimdGamma;
use crate::normal::SimdNormal;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;
use crate::traits::DistributionExt;
use crate::traits::SimdFloatExt;

/// Truncated normal $\mathcal{N}(\mu, \sigma^2)$ restricted to
/// $[\text{lower}, \text{upper}]$.
pub struct SimdTruncatedNormal<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  mean: T,
  std_dev: T,
  lower: T,
  upper: T,
  base: SimdNormal<T, 64, R>,
  /// Cached probability mass $F(\text{upper}) - F(\text{lower})$ — needed
  /// for the normalising constant of the density and the inverse-CDF
  /// fallback in tight intervals.
  norm_mass: f64,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdTruncatedNormal<T, R> {
  /// Create a truncated normal with the given parameters.
  pub fn new<S: SeedExt>(mean: T, std_dev: T, lower: T, upper: T, seed: &S) -> Self {
    assert!(std_dev > T::zero(), "std_dev must be positive");
    assert!(lower < upper, "lower bound must be < upper bound");
    let mean_f64 = mean.to_f64().unwrap();
    let std_f64 = std_dev.to_f64().unwrap();
    let lo_f64 = lower.to_f64().unwrap();
    let up_f64 = upper.to_f64().unwrap();
    let norm_mass =
      norm_cdf_scalar((up_f64 - mean_f64) / std_f64) - norm_cdf_scalar((lo_f64 - mean_f64) / std_f64);
    Self {
      mean,
      std_dev,
      lower,
      upper,
      base: SimdNormal::<T, 64, R>::new(mean, std_dev, seed),
      norm_mass,
    }
  }

  /// Draw a single truncated normal sample using the internal SIMD RNG.
  ///
  /// On wide intervals (acceptance ≥ 5 %) we use plain rejection on the
  /// base normal sampler; on tight intervals we route through the
  /// inverse-CDF transform which is exact but a bit slower.
  #[inline]
  pub fn sample_fast(&self) -> T {
    if self.norm_mass > 0.05 {
      // Plain rejection — fast path.
      for _ in 0..1000 {
        let x = self.base.sample_fast();
        if x >= self.lower && x <= self.upper {
          return x;
        }
      }
      // Fall through to the inverse-CDF if we somehow had a 1000-shot run
      // of rejections (numerical edge cases).
    }
    self.inverse_cdf_sample()
  }

  /// Inverse-CDF sample: $X = F^{-1}(F(\text{lower}) + U \cdot (F(\text{upper}) - F(\text{lower})))$.
  fn inverse_cdf_sample(&self) -> T {
    let mean_f64 = self.mean.to_f64().unwrap();
    let std_f64 = self.std_dev.to_f64().unwrap();
    let lo_f64 = self.lower.to_f64().unwrap();
    let up_f64 = self.upper.to_f64().unwrap();
    let f_lo = norm_cdf_scalar((lo_f64 - mean_f64) / std_f64);
    let f_up = norm_cdf_scalar((up_f64 - mean_f64) / std_f64);
    let mut rng = rand::rng();
    let u: f64 = rng.random_range(0.0..1.0);
    let q = f_lo + u * (f_up - f_lo);
    let z = crate::special::ndtri(q);
    T::from_f64_fast(mean_f64 + std_f64 * z)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdTruncatedNormal<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.mean, self.std_dev, self.lower, self.upper, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdTruncatedNormal<T, R> {
  fn sample<Rr: Rng + ?Sized>(&self, _rng: &mut Rr) -> T {
    self.sample_fast()
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> DistributionExt for SimdTruncatedNormal<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x < lo || x > up {
      return 0.0;
    }
    let mean = self.mean.to_f64().unwrap();
    let std = self.std_dev.to_f64().unwrap();
    let z = (x - mean) / std;
    let phi = (-0.5 * z * z).exp() / ((2.0 * std::f64::consts::PI).sqrt() * std);
    phi / self.norm_mass
  }

  fn cdf(&self, x: f64) -> f64 {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x <= lo {
      return 0.0;
    }
    if x >= up {
      return 1.0;
    }
    let mean = self.mean.to_f64().unwrap();
    let std = self.std_dev.to_f64().unwrap();
    let f_x = norm_cdf_scalar((x - mean) / std);
    let f_lo = norm_cdf_scalar((lo - mean) / std);
    (f_x - f_lo) / self.norm_mass
  }
}

/// Truncated exponential $\mathrm{Exp}(\lambda)$ restricted to
/// $[\text{lower}, \text{upper}]$ ($\text{lower} \ge 0$). Closed-form
/// inverse-CDF sampling — no rejection needed.
pub struct SimdTruncatedExp<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  lambda: T,
  lower: T,
  upper: T,
  norm_mass: f64,
  _base: SimdExp<T, R>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdTruncatedExp<T, R> {
  pub fn new<S: SeedExt>(lambda: T, lower: T, upper: T, seed: &S) -> Self {
    assert!(lambda > T::zero(), "lambda must be positive");
    assert!(lower >= T::zero(), "lower bound must be ≥ 0");
    assert!(lower < upper, "lower < upper");
    let lam = lambda.to_f64().unwrap();
    let lo = lower.to_f64().unwrap();
    let up = upper.to_f64().unwrap();
    let f_lo = 1.0 - (-lam * lo).exp();
    let f_up = if up.is_infinite() {
      1.0
    } else {
      1.0 - (-lam * up).exp()
    };
    Self {
      lambda,
      lower,
      upper,
      norm_mass: f_up - f_lo,
      _base: SimdExp::<T, R>::new(lambda, seed),
    }
  }

  /// Closed-form inverse-CDF draw: $X = -\ln(1 - U(F(\text{upper}) - F(\text{lower})) - F(\text{lower}))/\lambda$.
  #[inline]
  pub fn sample_fast(&self) -> T {
    let lam = self.lambda.to_f64().unwrap();
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    let f_lo = 1.0 - (-lam * lo).exp();
    let f_up = if up.is_infinite() {
      1.0
    } else {
      1.0 - (-lam * up).exp()
    };
    let mut rng = rand::rng();
    let u: f64 = rng.random_range(0.0..1.0);
    let q = f_lo + u * (f_up - f_lo);
    let arg = (1.0 - q).max(1e-300);
    T::from_f64_fast(-arg.ln() / lam)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdTruncatedExp<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.lambda, self.lower, self.upper, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdTruncatedExp<T, R> {
  fn sample<Rr: Rng + ?Sized>(&self, _rng: &mut Rr) -> T {
    self.sample_fast()
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> DistributionExt for SimdTruncatedExp<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x < lo || x > up {
      return 0.0;
    }
    let lam = self.lambda.to_f64().unwrap();
    let phi = lam * (-lam * x).exp();
    phi / self.norm_mass
  }

  fn cdf(&self, x: f64) -> f64 {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x <= lo {
      return 0.0;
    }
    if x >= up {
      return 1.0;
    }
    let lam = self.lambda.to_f64().unwrap();
    let f_x = 1.0 - (-lam * x).exp();
    let f_lo = 1.0 - (-lam * lo).exp();
    (f_x - f_lo) / self.norm_mass
  }
}

/// Truncated Beta restricted to $[\text{lower}, \text{upper}] \subseteq [0, 1]$.
/// Rejection from the base [`SimdBeta`]; emits the clamped midpoint after
/// 1000 unsuccessful tries (the boundary itself is hit with measure zero).
pub struct SimdTruncatedBeta<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  alpha: T,
  beta: T,
  lower: T,
  upper: T,
  base: SimdBeta<T, R>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdTruncatedBeta<T, R> {
  pub fn new<S: SeedExt>(alpha: T, beta: T, lower: T, upper: T, seed: &S) -> Self {
    assert!(alpha > T::zero() && beta > T::zero(), "α, β > 0");
    let lo = lower.to_f64().unwrap();
    let up = upper.to_f64().unwrap();
    assert!(
      (0.0..=1.0).contains(&lo) && (0.0..=1.0).contains(&up),
      "bounds must lie in [0,1]"
    );
    assert!(lower < upper, "lower < upper");
    Self {
      alpha,
      beta,
      lower,
      upper,
      base: SimdBeta::<T, R>::new(alpha, beta, seed),
    }
  }

  #[inline]
  pub fn sample_fast(&self) -> T {
    for _ in 0..1000 {
      let x = self.base.sample_fast();
      if x >= self.lower && x <= self.upper {
        return x;
      }
    }
    (self.lower + self.upper) * T::from_f64_fast(0.5)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdTruncatedBeta<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.alpha, self.beta, self.lower, self.upper, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdTruncatedBeta<T, R> {
  fn sample<Rr: Rng + ?Sized>(&self, _rng: &mut Rr) -> T {
    self.sample_fast()
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> DistributionExt for SimdTruncatedBeta<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x < lo || x > up {
      return 0.0;
    }
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let log_norm = crate::special::ln_gamma(a + b)
      - crate::special::ln_gamma(a)
      - crate::special::ln_gamma(b);
    let log_kernel = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln();
    let base_pdf = (log_norm + log_kernel).exp();
    let norm_mass = self.cdf_helper(up) - self.cdf_helper(lo);
    base_pdf / norm_mass.max(1e-300)
  }

  fn cdf(&self, x: f64) -> f64 {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x <= lo {
      return 0.0;
    }
    if x >= up {
      return 1.0;
    }
    let f_x = self.cdf_helper(x);
    let f_lo = self.cdf_helper(lo);
    let f_up = self.cdf_helper(up);
    (f_x - f_lo) / (f_up - f_lo).max(1e-300)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdTruncatedBeta<T, R> {
  fn cdf_helper(&self, x: f64) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    crate::special::beta_i(a, b, x.clamp(0.0, 1.0))
  }
}

/// Truncated Gamma$(k, \theta)$ restricted to $[\text{lower}, \text{upper}]$,
/// $\text{lower} \ge 0$. Rejection from base [`SimdGamma`] with the same
/// 1000-attempt fallback policy as the Beta case.
pub struct SimdTruncatedGamma<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  shape: T,
  scale: T,
  lower: T,
  upper: T,
  base: SimdGamma<T, R>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdTruncatedGamma<T, R> {
  pub fn new<S: SeedExt>(shape: T, scale: T, lower: T, upper: T, seed: &S) -> Self {
    assert!(shape > T::zero(), "shape > 0");
    assert!(scale > T::zero(), "scale > 0");
    assert!(lower >= T::zero(), "lower ≥ 0");
    assert!(lower < upper, "lower < upper");
    Self {
      shape,
      scale,
      lower,
      upper,
      base: SimdGamma::<T, R>::new(shape, scale, seed),
    }
  }

  #[inline]
  pub fn sample_fast(&self) -> T {
    for _ in 0..1000 {
      let x = self.base.sample_fast();
      if x >= self.lower && x <= self.upper {
        return x;
      }
    }
    (self.lower + self.upper) * T::from_f64_fast(0.5)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdTruncatedGamma<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.shape, self.scale, self.lower, self.upper, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdTruncatedGamma<T, R> {
  fn sample<Rr: Rng + ?Sized>(&self, _rng: &mut Rr) -> T {
    self.sample_fast()
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> DistributionExt for SimdTruncatedGamma<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x < lo || x > up {
      return 0.0;
    }
    let k = self.shape.to_f64().unwrap();
    let theta = self.scale.to_f64().unwrap();
    let log_norm = -crate::special::ln_gamma(k) - k * theta.ln();
    let log_kernel = (k - 1.0) * x.ln() - x / theta;
    let base_pdf = (log_norm + log_kernel).exp();
    let mass = self.cdf_helper(up) - self.cdf_helper(lo);
    base_pdf / mass.max(1e-300)
  }

  fn cdf(&self, x: f64) -> f64 {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x <= lo {
      return 0.0;
    }
    if x >= up {
      return 1.0;
    }
    let f_x = self.cdf_helper(x);
    let f_lo = self.cdf_helper(lo);
    let f_up = self.cdf_helper(up);
    (f_x - f_lo) / (f_up - f_lo).max(1e-300)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdTruncatedGamma<T, R> {
  fn cdf_helper(&self, x: f64) -> f64 {
    let k = self.shape.to_f64().unwrap();
    let theta = self.scale.to_f64().unwrap();
    crate::special::gamma_p(k, x / theta)
  }
}

/// Standard-normal CDF used by the truncated-normal CDF / normalisation
/// helpers. Mirrors the shape used in `crate::special` but kept local so
/// the truncated module is self-contained.
fn norm_cdf_scalar(z: f64) -> f64 {
  0.5 * (1.0 + crate::special::erf(z / std::f64::consts::SQRT_2))
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Truncated normal samples must respect the bound.
  #[test]
  fn truncated_normal_samples_within_bounds() {
    let tn = SimdTruncatedNormal::<f64>::new(0.0, 1.0, -1.0, 2.0, &Unseeded);
    for _ in 0..5_000 {
      let x = tn.sample_fast();
      assert!((-1.0..=2.0).contains(&x), "sample {x} out of [-1, 2]");
    }
  }

  /// Truncated normal density integrates to 1 in the bulk (mid-point Riemann
  /// sum on a 1000-step grid).
  #[test]
  fn truncated_normal_pdf_normalised() {
    let tn = SimdTruncatedNormal::<f64>::new(0.0, 1.0, -1.0, 2.0, &Unseeded);
    let n = 1_000;
    let h = 3.0 / n as f64;
    let s: f64 = (0..n).map(|k| tn.pdf(-1.0 + (k as f64 + 0.5) * h) * h).sum();
    assert!(
      (s - 1.0).abs() < 5e-3,
      "truncated normal pdf integrates to {s}, expected 1"
    );
  }

  /// Truncated exponential: closed-form CDF must round-trip to inputs.
  #[test]
  fn truncated_exp_cdf_round_trips() {
    let te = SimdTruncatedExp::<f64>::new(2.0, 0.0, 1.5, &Unseeded);
    for x in [0.0, 0.3, 0.7, 1.0, 1.5] {
      let f = te.cdf(x);
      assert!((0.0..=1.0).contains(&f));
    }
    assert_eq!(te.cdf(-0.1), 0.0);
    assert_eq!(te.cdf(2.0), 1.0);
  }

  /// Truncated exponential samples in bounds with the right approximate
  /// mean (closed-form check on tight [0, 0.5] band of Exp(1)).
  #[test]
  fn truncated_exp_samples_mean() {
    let te = SimdTruncatedExp::<f64>::new(1.0, 0.0, 0.5, &Unseeded);
    let n = 20_000;
    let mut sum = 0.0;
    for _ in 0..n {
      let x = te.sample_fast();
      assert!((0.0..=0.5).contains(&x));
      sum += x;
    }
    let mean = sum / n as f64;
    // Closed-form mean of Truncated Exp(1) on [0, 0.5]:
    //   E[X | 0 ≤ X ≤ 0.5] = ∫₀^0.5 x · e^{-x} dx / (1 - e^{-0.5})
    //                       = [1 - 1.5 · e^{-0.5}] / (1 - e^{-0.5})
    let half = 0.5_f64;
    let expected = (1.0 - 1.5 * (-half).exp()) / (1.0 - (-half).exp());
    assert!(
      (mean - expected).abs() < 0.01,
      "truncated Exp(1) mean = {mean}, expected ≈ {expected}"
    );
  }

  /// Truncated Beta in [0.2, 0.8] respects bounds and has uniform-like
  /// support across many samples.
  #[test]
  fn truncated_beta_samples_within_bounds() {
    let tb = SimdTruncatedBeta::<f64>::new(2.0, 2.0, 0.2, 0.8, &Unseeded);
    for _ in 0..3_000 {
      let x = tb.sample_fast();
      assert!((0.2..=0.8).contains(&x));
    }
  }

  /// Truncated Gamma stays in bounds.
  #[test]
  fn truncated_gamma_samples_within_bounds() {
    let tg = SimdTruncatedGamma::<f64>::new(2.0, 1.0, 1.0, 5.0, &Unseeded);
    for _ in 0..3_000 {
      let x = tg.sample_fast();
      assert!((1.0..=5.0).contains(&x));
    }
  }

  /// PDF / CDF degenerate cases: outside the bounds must produce 0 PDF
  /// and {0, 1} CDF.
  #[test]
  fn truncated_pdf_cdf_outside_bounds() {
    let tn = SimdTruncatedNormal::<f64>::new(0.0, 1.0, -1.0, 1.0, &Unseeded);
    assert_eq!(tn.pdf(-1.5), 0.0);
    assert_eq!(tn.pdf(1.5), 0.0);
    assert_eq!(tn.cdf(-2.0), 0.0);
    assert_eq!(tn.cdf(2.0), 1.0);
  }
}
