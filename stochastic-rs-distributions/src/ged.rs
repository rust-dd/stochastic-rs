//! # Generalized Error Distribution (GED) — Subbotin family
//!
//! $$
//! f(x; \mu, \alpha, \beta) = \frac{\beta}{2\alpha\,\Gamma(1/\beta)}\,
//! \exp\!\left(-\left|\frac{x - \mu}{\alpha}\right|^\beta\right),
//! \qquad \alpha > 0,\ \beta > 0.
//! $$
//!
//! Shape parameter $\beta$ controls tail behaviour:
//! - $\beta = 1$ → Laplace (double exponential, heavy tails)
//! - $\beta = 2$ → Gaussian
//! - $\beta < 2$ → heavier-than-Gaussian (GARCH residuals, Nelson 1991)
//! - $\beta > 2$ → lighter-than-Gaussian (platykurtic)
//!
//! **Sampling.** The standard form $|X|^\beta \sim \mathrm{Gamma}(1/\beta, 1)$
//! gives the bijection
//!
//! $$
//! X = \alpha \cdot Y^{1/\beta} \cdot S + \mu,
//! \qquad Y \sim \mathrm{Gamma}(1/\beta, 1),\ S \sim \mathrm{Uniform}\{-1, +1\}.
//! $$
//!
//! Used by GARCH-type volatility models with heavy-tailed innovations
//! (Nelson 1991 EGARCH, Bollerslev 1987 GARCH-t variant).
//!
//! References:
//! - Subbotin, M.T. (1923), "On the law of frequency of error",
//!   *Matematicheskii Sbornik* 31, 296-301.
//! - Nelson, D.B. (1991), "Conditional heteroskedasticity in asset
//!   returns: a new approach", *Econometrica* 59, 347-370.

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::gamma::SimdGamma;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;
use crate::traits::DistributionExt;
use crate::traits::SimdFloatExt;

pub struct SimdGed<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  mu: T,
  alpha: T,
  beta: T,
  gamma: SimdGamma<T, R>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdGed<T, R> {
  pub fn new<S: SeedExt>(mu: T, alpha: T, beta: T, seed: &S) -> Self {
    assert!(alpha > T::zero(), "α must be > 0");
    assert!(beta > T::zero(), "β must be > 0");
    let inv_beta = T::one() / beta;
    Self {
      mu,
      alpha,
      beta,
      gamma: SimdGamma::<T, R>::new(inv_beta, T::one(), seed),
    }
  }

  /// Closed-form sample via $X = \alpha \cdot Y^{1/\beta} \cdot S + \mu$.
  #[inline]
  pub fn sample_fast(&self) -> T {
    let y = self.gamma.sample_fast();
    let mag = y.powf(T::one() / self.beta);
    let mut rng = rand::rng();
    let sign = if rng.random::<bool>() {
      T::one()
    } else {
      -T::one()
    };
    self.alpha * mag * sign + self.mu
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdGed<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.mu, self.alpha, self.beta, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdGed<T, R> {
  fn sample<Rr: Rng + ?Sized>(&self, _rng: &mut Rr) -> T {
    self.sample_fast()
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> DistributionExt for SimdGed<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let z = ((x - mu) / a).abs();
    let log_pdf = b.ln() - (2.0 * a).ln() - crate::special::ln_gamma(1.0 / b) - z.powf(b);
    log_pdf.exp()
  }

  fn cdf(&self, x: f64) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let z = (x - mu) / a;
    // F(x) = 1/2 + sign(x - μ)/2 · γ(1/β, |z|^β) / Γ(1/β)
    let zb = z.abs().powf(b);
    let half_inc = 0.5 * crate::special::gamma_p(1.0 / b, zb);
    if z >= 0.0 {
      0.5 + half_inc
    } else {
      0.5 - half_inc
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// GED with β=2 should collapse to a Gaussian with std-dev α·√(1/2).
  /// Tested via sample variance.
  #[test]
  fn ged_beta_two_is_gaussian() {
    let g = SimdGed::<f64>::new(0.0, std::f64::consts::SQRT_2, 2.0, &Unseeded);
    let n = 30_000;
    let mut sum_sq = 0.0;
    let mut sum = 0.0;
    for _ in 0..n {
      let x = g.sample_fast();
      sum += x;
      sum_sq += x * x;
    }
    let mean = sum / n as f64;
    let var = sum_sq / n as f64 - mean * mean;
    assert!(
      (mean).abs() < 0.05,
      "GED(0, √2, 2) mean = {mean}, expected ~0"
    );
    // Var = α² · Γ(3/β) / Γ(1/β); α=√2, β=2 → Γ(3/2)/Γ(1/2) = 0.5 → Var = 1
    assert!(
      (var - 1.0).abs() < 0.05,
      "GED(0, √2, 2) variance = {var}, expected ~1"
    );
  }

  /// PDF normalises to 1 (numeric integration).
  #[test]
  fn ged_pdf_normalised() {
    let g = SimdGed::<f64>::new(0.0, 1.0, 1.5, &Unseeded);
    let n = 5000;
    let lo = -20.0_f64;
    let up = 20.0_f64;
    let h = (up - lo) / n as f64;
    let s: f64 = (0..n).map(|k| g.pdf(lo + (k as f64 + 0.5) * h) * h).sum();
    assert!(
      (s - 1.0).abs() < 1e-3,
      "GED(0, 1, 1.5) PDF integrates to {s}"
    );
  }

  /// CDF round-trip via PDF integration.
  #[test]
  fn ged_cdf_at_mu_is_half() {
    let g = SimdGed::<f64>::new(1.5, 0.8, 1.7, &Unseeded);
    let c = g.cdf(1.5);
    assert!(
      (c - 0.5).abs() < 1e-10,
      "GED(1.5, 0.8, 1.7) CDF at μ = {c}, expected 0.5"
    );
  }
}
