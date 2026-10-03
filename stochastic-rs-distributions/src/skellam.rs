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
//! two-Poisson subtraction; the PMF and CDF go through the scaled modified
//! Bessel function [`crate::special::ln_bessel_ie`].
//!
//! Reference: Skellam, J.G. (1946), "The frequency distribution of the
//! difference between two Poisson variates belonging to different
//! populations", *Journal of the Royal Statistical Society* 109(3), 296,
//! DOI 10.2307/2981372.

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use crate::poisson::SimdPoisson;
use crate::seeded::StreamState;
use crate::special::ln_bessel_ie;
use crate::traits::DistributionExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// Skellam law of `N₁ − N₂` for independent Poisson counts of rates `mu1` and `mu2`: parameters only; a
/// [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Debug, PartialEq)]
pub struct SimdSkellam {
  mu1: f64,
  mu2: f64,
  p1: SimdPoisson<u32>,
  p2: SimdPoisson<u32>,
}

impl SimdSkellam {
  /// Construct a Skellam$(\mu_1, \mu_2)$ random variable.
  ///
  /// - `mu1` — finite rate μ₁ > 0 of the minuend Poisson N₁ (matches the module
  ///   header's μ₁).
  /// - `mu2` — finite rate μ₂ > 0 of the subtrahend Poisson N₂ (matches the
  ///   module header's μ₂). The output is `N₁ - N₂`.
  pub fn new(mu1: f64, mu2: f64) -> Self {
    assert!(
      mu1 > 0.0 && mu1.is_finite(),
      "mu1 must satisfy `mu1 > 0.0 && mu1.is_finite()`, got mu1 = {mu1:?}"
    );
    assert!(
      mu2 > 0.0 && mu2.is_finite(),
      "mu2 must satisfy `mu2 > 0.0 && mu2.is_finite()`, got mu2 = {mu2:?}"
    );
    Self {
      mu1,
      mu2,
      p1: SimdPoisson::new(mu1),
      p2: SimdPoisson::new(mu2),
    }
  }

  /// The minuend rate `μ₁`.
  pub fn mu1(&self) -> f64 {
    self.mu1
  }

  /// The subtrahend rate `μ₂`.
  pub fn mu2(&self) -> f64 {
    self.mu2
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> i64 {
    let n1 = self.p1.draw_with(rng);
    let n2 = self.p2.draw_with(rng);
    n1 as i64 - n2 as i64
  }
}

impl Sealed for SimdSkellam {}

impl SimdDistribution for SimdSkellam {
  type State<R: SimdRngExt> = (StreamState<u32, R, 16>, StreamState<u32, R, 16>);

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (Self::State<R>, u64) {
    let (p1, basis) = self.p1.init::<R, S>(seed);
    let (p2, _) = self.p2.init::<R, S>(seed);
    ((p1, p2), basis)
  }
}

impl SimdKernel for SimdSkellam {
  type Item = i64;

  fn fill<R: SimdRngExt>(&self, state: &mut Self::State<R>, out: &mut [i64]) {
    for x in out.iter_mut() {
      *x = self.next(state);
    }
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut Self::State<R>) -> i64 {
    let n1 = self.p1.next(&mut state.0);
    let n2 = self.p2.next(&mut state.1);
    n1 as i64 - n2 as i64
  }
}

impl Distribution<i64> for SimdSkellam {
  /// Two Poisson table inversions, `N₁` then `N₂`, on the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> i64 {
    self.draw_with(rng)
  }
}

impl DistributionExt for SimdSkellam {
  /// PMF $P(X = k)$. The argument is a float by convention; only the
  /// rounded integer part is meaningful.
  fn pdf(&self, x: f64) -> f64 {
    let k = x.round();
    let gap = self.mu1.sqrt() - self.mu2.sqrt();
    let z = 2.0 * (self.mu1 * self.mu2).sqrt();
    (0.5 * k * (self.mu1 / self.mu2).ln() - gap * gap + ln_bessel_ie(k.abs(), z)).exp()
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

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::*;
  use crate::tests::scalar_chi_square_best_p;

  /// An infinite rate is named by its own argument before it reaches the Poisson table.
  #[test]
  #[should_panic(expected = "mu2 must satisfy `mu2 > 0.0 && mu2.is_finite()`, got mu2 = inf")]
  fn skellam_infinite_rate_is_rejected() {
    SimdSkellam::new(1.0, f64::INFINITY);
  }

  /// Mean and variance match $\mu_1 - \mu_2$ and $\mu_1 + \mu_2$ within
  /// 3σ on 30k samples.
  #[test]
  fn skellam_sample_moments() {
    let mut s = SimdSkellam::new(3.0, 2.0).seeded(&Unseeded);
    let n = 30_000;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for _ in 0..n {
      let x = s.sample() as f64;
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
    let s = SimdSkellam::new(2.0, 2.0);
    let mut total = 0.0;
    for k in -30..=30 {
      total += s.pdf(k as f64);
    }
    assert!(
      (total - 1.0).abs() < 1e-6,
      "Skellam(2, 2) PMF sum = {total}, expected ≈ 1"
    );
  }

  /// Exact pmf from `mpmath.besseli` at 60 digits; the old 200-term series collapsed
  /// once μ₁ + μ₂ reached a few hundred.
  #[test]
  fn skellam_pmf_matches_mpmath_at_large_rates() {
    for ((mu1, mu2, k), want) in [
      ((2.0, 2.0, 0.0), 0.207_001_921_223_986_7),
      ((2.0, 1.5, -3.0), 0.034_258_044_673_158_364),
      ((50.0, 40.0, 10.0), 0.042_109_756_037_380_416),
      ((400.0, 300.0, 100.0), 0.015_081_203_817_903_906),
      ((400.0, 300.0, 40.0), 0.001_147_244_606_054_886_6),
    ] {
      let got = SimdSkellam::new(mu1, mu2).pdf(k);
      assert!(
        ((got - want) / want).abs() < 1e-12,
        "Skellam({mu1}, {mu2}) pmf({k}) = {got}, want {want}"
      );
    }
  }

  /// CDF at the right tail must reach 1.
  #[test]
  fn skellam_cdf_tail_unity() {
    let s = SimdSkellam::new(2.0, 1.5);
    let c = s.cdf(50.0);
    assert!((c - 1.0).abs() < 1e-6, "Skellam CDF at +∞ ≈ {c}");
  }

  /// The honest `Distribution` draws from the caller's rng and agrees with the cdf.
  #[test]
  fn scalar_sample_matches_cdf() {
    let d = SimdSkellam::new(9.0, 5.0);
    let best = scalar_chi_square_best_p(&d, (-26, 34), |k| d.cdf(k as f64));
    assert!(best > 0.01, "best p = {best}");
  }
}
