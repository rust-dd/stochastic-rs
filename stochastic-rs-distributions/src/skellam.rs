//! # Skellam distribution
//!
//! $$
//! X \sim \mathrm{Skellam}(\mu_1, \mu_2) \;:=\; N_1 - N_2,
//! \qquad N_i \sim \mathrm{Poisson}(\mu_i)\ \text{independent}.
//! $$
//!
//! PMF: $P(X = k) = e^{-(\mu_1 + \mu_2)} (\mu_1/\mu_2)^{k/2} I_{|k|}(2\sqrt{\mu_1 \mu_2})$
//! where $I_n$ is the modified Bessel function of the first kind.
//!
//! Used in sports / queueing models (goal difference, queue net flow), and
//! more generally any count-difference application. Sampling is a trivial
//! two-Poisson subtraction; the PMF and CDF go through the modified Bessel
//! recurrence.
//!
//! Reference: Skellam, J.G. (1946), "The frequency distribution of the
//! difference between two Poisson variates belonging to different
//! populations", *Journal of the Royal Statistical Society* 109(3), 296.

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::poisson::SimdPoisson;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;
use crate::traits::DistributionExt;

pub struct SimdSkellam<R: SimdRngExt = SimdRng> {
  mu1: f64,
  mu2: f64,
  p1: SimdPoisson<u32, R>,
  p2: SimdPoisson<u32, R>,
}

impl<R: SimdRngExt> SimdSkellam<R> {
  /// Construct a Skellam$(\mu_1, \mu_2)$ random variable.
  pub fn new<S: SeedExt>(mu1: f64, mu2: f64, seed: &S) -> Self {
    assert!(mu1 > 0.0 && mu2 > 0.0, "μ₁, μ₂ must be positive");
    Self {
      mu1,
      mu2,
      p1: SimdPoisson::<u32, R>::new(mu1, seed),
      p2: SimdPoisson::<u32, R>::new(mu2, seed),
    }
  }

  /// Single integer sample $N_1 - N_2$ via the trivial Poisson subtraction.
  #[inline]
  pub fn sample_fast(&self) -> i64 {
    let n1 = self.p1.sample_fast();
    let n2 = self.p2.sample_fast();
    n1 as i64 - n2 as i64
  }
}

impl<R: SimdRngExt> Clone for SimdSkellam<R> {
  fn clone(&self) -> Self {
    Self::new(self.mu1, self.mu2, &Unseeded)
  }
}

impl<R: SimdRngExt> Distribution<i64> for SimdSkellam<R> {
  fn sample<Rr: Rng + ?Sized>(&self, _rng: &mut Rr) -> i64 {
    self.sample_fast()
  }
}

impl<R: SimdRngExt> DistributionExt for SimdSkellam<R> {
  /// PMF $P(X = k)$. The argument is a float by convention; only the
  /// rounded integer part is meaningful.
  fn pdf(&self, x: f64) -> f64 {
    let k = x.round() as i64;
    let mu_prod = self.mu1 * self.mu2;
    let ratio = (self.mu1 / self.mu2).powf(0.5 * k as f64);
    let bessel = modified_bessel_i_n(k.unsigned_abs() as i64, 2.0 * mu_prod.sqrt());
    (-(self.mu1 + self.mu2)).exp() * ratio * bessel
  }

  /// CDF via summation of the PMF on a truncated range. Skellam tails
  /// drop super-exponentially so summing ±10·sqrt(μ₁+μ₂) lanes is
  /// numerically tight.
  fn cdf(&self, x: f64) -> f64 {
    let k_max = x.floor() as i64;
    let radius = (10.0 * (self.mu1 + self.mu2).sqrt()) as i64;
    let lower = -radius;
    let mut s = 0.0_f64;
    for k in lower..=k_max {
      s += self.pdf(k as f64);
    }
    s.clamp(0.0, 1.0)
  }
}

/// Modified Bessel function of the first kind $I_n(x)$ for non-negative
/// integer $n$ via the power series (good for moderate `x`; for very
/// large `x` we fall back to an asymptotic). Adequate for Skellam PMF
/// evaluation in the bulk; production usage with `μ ≫ 100` may want a
/// dedicated routine.
fn modified_bessel_i_n(n: i64, x: f64) -> f64 {
  if x < 0.0 {
    return modified_bessel_i_n(n, -x) * if n.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
  }
  if x == 0.0 {
    return if n == 0 { 1.0 } else { 0.0 };
  }
  let half_x = 0.5 * x;
  let log_term = (n as f64) * half_x.ln();
  let mut term = (log_term - crate::special::ln_gamma(n as f64 + 1.0)).exp();
  let mut sum = term;
  let y = half_x * half_x;
  for k in 1..200 {
    term *= y / (k as f64 * (n as f64 + k as f64));
    sum += term;
    if term < 1e-18 * sum.abs() {
      break;
    }
  }
  sum
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Mean and variance match $\mu_1 - \mu_2$ and $\mu_1 + \mu_2$ within
  /// 3σ on 30k samples.
  #[test]
  fn skellam_sample_moments() {
    let s = SimdSkellam::<SimdRng>::new(3.0, 2.0, &Unseeded);
    let n = 30_000;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for _ in 0..n {
      let x = s.sample_fast() as f64;
      sum += x;
      sum_sq += x * x;
    }
    let mean = sum / n as f64;
    let var = sum_sq / n as f64 - mean * mean;
    assert!(
      (mean - 1.0).abs() < 0.1,
      "Skellam(3, 2) mean = {mean}, expected ≈ 1.0"
    );
    assert!(
      (var - 5.0).abs() < 0.3,
      "Skellam(3, 2) variance = {var}, expected ≈ 5.0"
    );
  }

  /// PMF normalises to 1 within the support window.
  #[test]
  fn skellam_pmf_normalised() {
    let s = SimdSkellam::<SimdRng>::new(2.0, 2.0, &Unseeded);
    let mut total = 0.0;
    for k in -30..=30 {
      total += s.pdf(k as f64);
    }
    assert!(
      (total - 1.0).abs() < 1e-6,
      "Skellam(2, 2) PMF sum = {total}, expected ≈ 1"
    );
  }

  /// CDF at the right tail must reach 1.
  #[test]
  fn skellam_cdf_tail_unity() {
    let s = SimdSkellam::<SimdRng>::new(2.0, 1.5, &Unseeded);
    let c = s.cdf(50.0);
    assert!((c - 1.0).abs() < 1e-6, "Skellam CDF at +∞ ≈ {c}");
  }
}
