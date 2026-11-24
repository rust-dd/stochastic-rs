//! Margrabe's exchange option in closed form and the spread option of any strike by Monte Carlo.
//!
//! $$
//! M = S_1 e^{-q_1 T} N(d_1) - S_2 e^{-q_2 T} N(d_2),\quad
//! \sigma = \sqrt{\sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2}
//! $$
//!
//! Kirk's approximation for a nonzero strike is [`super::kirk`].
//! Reference: Margrabe (1978), J. Finance 33(1), 177–186, DOI 10.1111/j.1540-6261.1978.tb03397.x
//! Reference: Kirk (1995), "Correlation in the Energy Markets", Managing Energy Price Risk, 71–78

use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_stochastic::mc::McEstimate;

use crate::OptionType;
use crate::pricing::digital::expired;
use crate::pricing::digital::not_positive;
use crate::pricing::digital::vanilla_payoff;
use crate::pricing::mc_stats::off_domain;
use crate::pricing::mc_stats::par_estimate;
use crate::pricing::mc_stats::pathless;
use crate::traits::RealExt;

/// The two volatilities and the correlation every two-asset pricer takes.
#[track_caller]
pub(crate) fn assert_two_asset_model(sigma1: f64, sigma2: f64, rho: f64) {
  assert!(
    sigma1.is_finite() && sigma1 >= 0.0,
    "sigma1 must satisfy `sigma1.is_finite() && sigma1 >= 0.0`, got sigma1 = {sigma1:?}"
  );
  assert!(
    sigma2.is_finite() && sigma2 >= 0.0,
    "sigma2 must satisfy `sigma2.is_finite() && sigma2 >= 0.0`, got sigma2 = {sigma2:?}"
  );
  assert!(
    (-1.0..=1.0).contains(&rho),
    "rho must satisfy `(-1.0..=1.0).contains(&rho)`, got rho = {rho:?}"
  );
}

/// Margrabe (1978) exchange option, `max(S₁ − S₂, 0)`, with no rate (its two discounts cancel) and
/// no [`ModelPricer`](crate::traits::ModelPricer), which prices one underlying against a strike.
///
/// $$
/// V = S_1 e^{-q_1 T} N(d_1) - S_2 e^{-q_2 T} N(d_2),\quad
/// \sigma = \sqrt{\sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2}
/// $$
///
/// ```
/// use stochastic_rs_quant::pricing::spread::MargrabePricer;
///
/// let model = MargrabePricer::new(0.25, 0.20, 0.4);
/// let atm = model.price(100.0, 100.0, 0.0, 0.0, 1.0);
/// let itm = model.price(110.0, 100.0, 0.0, 0.0, 1.0);
/// assert!(itm > atm, "the right to exchange gains value as S1 rises");
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MargrabePricer {
  /// Volatility of asset 1.
  pub sigma1: f64,
  /// Volatility of asset 2.
  pub sigma2: f64,
  /// Correlation between log-returns.
  pub rho: f64,
}

impl MargrabePricer {
  /// Panics unless `sigma1`, `sigma2` are finite and `≥ 0` and `rho ∈ [-1, 1]`: at `rho = 5` the
  /// variance goes negative and the degenerate branch would return the discounted intrinsic.
  pub fn new(sigma1: f64, sigma2: f64, rho: f64) -> Self {
    assert_two_asset_model(sigma1, sigma2, rho);
    Self {
      sigma1,
      sigma2,
      rho,
    }
  }

  /// Combined exchange variance
  /// $\sigma^2 = \sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2$.
  fn combined_variance(&self) -> f64 {
    self.sigma1 * self.sigma1 + self.sigma2 * self.sigma2
      - 2.0 * self.rho * self.sigma1 * self.sigma2
  }

  /// $(d_1, d_2)$ at one query point, given the model's combined variance.
  fn d1_d2(v_sq: f64, s1: f64, s2: f64, q1: f64, q2: f64, tau: f64) -> (f64, f64) {
    let v = v_sq.sqrt();
    let sqrt_t = tau.sqrt();
    let d1 = ((s1 / s2).ln() + (q2 - q1 + 0.5 * v_sq) * tau) / (v * sqrt_t);
    (d1, d1 - v * sqrt_t)
  }

  /// Exchange-option price $\max(S_1 - S_2, 0)$ (swap the legs for the other side): the payoff at
  /// `tau = 0`, NaN past expiry or for a spot that is not positive before it.
  pub fn price(&self, s1: f64, s2: f64, q1: f64, q2: f64, tau: f64) -> f64 {
    if let Some(value) = expired(tau, &[q1, q2], || vanilla_payoff(OptionType::Call, s1, s2)) {
      return value;
    }
    if not_positive(&[s1, s2]) {
      return f64::NAN;
    }
    let v_sq = self.combined_variance();
    if v_sq < 1e-14 {
      return (s1 * (-q1 * tau).exp() - s2 * (-q2 * tau).exp()).max_or_nan(0.0);
    }
    let (d1, d2) = Self::d1_d2(v_sq, s1, s2, q1, q2, tau);
    s1 * (-q1 * tau).exp() * norm_cdf(d1) - s2 * (-q2 * tau).exp() * norm_cdf(d2)
  }

  /// Delta with respect to $S_1$; NaN at and past expiry and for a spot that is not positive.
  pub fn delta1(&self, s1: f64, s2: f64, q1: f64, q2: f64, tau: f64) -> f64 {
    if tau.is_nan() || tau <= 0.0 || not_positive(&[s1, s2]) {
      return f64::NAN;
    }
    let v_sq = self.combined_variance();
    if v_sq < 1e-14 {
      return (-q1 * tau).exp();
    }
    let (d1, _) = Self::d1_d2(v_sq, s1, s2, q1, q2, tau);
    (-q1 * tau).exp() * norm_cdf(d1)
  }

  /// Delta with respect to $S_2$; NaN at and past expiry and for a spot that is not positive.
  pub fn delta2(&self, s1: f64, s2: f64, q1: f64, q2: f64, tau: f64) -> f64 {
    if tau.is_nan() || tau <= 0.0 || not_positive(&[s1, s2]) {
      return f64::NAN;
    }
    let v_sq = self.combined_variance();
    if v_sq < 1e-14 {
      return -(-q2 * tau).exp();
    }
    let (_, d2) = Self::d1_d2(v_sq, s1, s2, q1, q2, tau);
    -(-q2 * tau).exp() * norm_cdf(d2)
  }
}

/// Monte Carlo spread option `max(φ(S₁ − S₂ − K), 0)` on correlated Gbm at any strike; each leg is
/// its own simulation, not parity off the other's sample.
///
/// ```
/// use stochastic_rs_quant::pricing::spread::McSpreadPricer;
///
/// let model = McSpreadPricer::new(0.25, 0.20, 0.4, 20_000);
/// let atm = model.price_call(110.0, 100.0, 10.0, 0.02, 0.0, 0.0, 1.0);
/// assert!(atm.mean > 0.0);
/// assert!(atm.std_err > 0.0);
/// println!("{atm}"); // e.g. "11.2 ± 0.05"
/// ```
#[derive(Debug, Clone, Copy)]
pub struct McSpreadPricer<S: SeedExt = Unseeded> {
  /// Volatility of asset 1.
  sigma1: f64,
  /// Volatility of asset 2.
  sigma2: f64,
  /// Correlation.
  rho: f64,
  /// Number of MC paths.
  n_paths: usize,
  /// Each call draws from a clone, so a `Deterministic` seed replays.
  pub seed: S,
}

impl McSpreadPricer {
  /// Panics unless `sigma1`, `sigma2` are finite and `≥ 0`, `rho ∈ [-1, 1]` (the factor
  /// `√(1 − ρ²)` would absorb it) and `n_paths >= 1`.
  pub fn new(sigma1: f64, sigma2: f64, rho: f64, n_paths: usize) -> Self {
    assert_two_asset_model(sigma1, sigma2, rho);
    assert!(
      n_paths >= 1,
      "n_paths must satisfy `n_paths >= 1`, got n_paths = {n_paths:?}"
    );
    Self {
      sigma1,
      sigma2,
      rho,
      n_paths,
      seed: Unseeded,
    }
  }
}

impl<S: SeedExt> McSpreadPricer<S> {
  pub fn with_seed<S2: SeedExt>(self, seed: S2) -> McSpreadPricer<S2> {
    McSpreadPricer {
      sigma1: self.sigma1,
      sigma2: self.sigma2,
      rho: self.rho,
      n_paths: self.n_paths,
      seed,
    }
  }

  /// Either leg at one query point from one simulation; NaN for a NaN query, `tau < 0` or a spot
  /// that is not positive before expiry.
  // One argument per term of the query.
  #[allow(clippy::too_many_arguments)]
  pub fn price_option(
    &self,
    s1: f64,
    s2: f64,
    k: f64,
    r: f64,
    q1: f64,
    q2: f64,
    tau: f64,
    option_type: OptionType,
  ) -> McEstimate<f64> {
    if off_domain(&[s1, s2], &[k, r, q1, q2], tau) {
      return pathless(f64::NAN);
    }
    let phi = match option_type {
      OptionType::Call => 1.0,
      OptionType::Put => -1.0,
    };
    let drift1 = (r - q1 - 0.5 * self.sigma1 * self.sigma1) * tau;
    let drift2 = (r - q2 - 0.5 * self.sigma2 * self.sigma2) * tau;
    let vol1 = self.sigma1 * tau.sqrt();
    let vol2 = self.sigma2 * tau.sqrt();
    let rho = self.rho;
    let sqrt_one_minus_rho2 = (1.0 - rho * rho).max(0.0).sqrt();

    let mut all_z = vec![0.0_f64; self.n_paths * 2];
    let seed = self.seed.clone();
    SimdNormal::<f64>::new(0.0, 1.0)
      .seeded(&seed)
      .fill_slice(&mut all_z);

    let payoff_of = |i: usize| -> f64 {
      let z1 = all_z[2 * i];
      let z2_indep = all_z[2 * i + 1];
      let z2 = rho * z1 + sqrt_one_minus_rho2 * z2_indep;
      let s1_t = s1 * (drift1 + vol1 * z1).exp();
      let s2_t = s2 * (drift2 + vol2 * z2).exp();
      (phi * (s1_t - s2_t - k)).max_or_nan(0.0)
    };
    par_estimate(self.n_paths, (-r * tau).exp(), payoff_of)
  }

  /// Price the spread call $\max(S_1-S_2-K,0)$ at one query point.
  // One argument per term of the query.
  #[allow(clippy::too_many_arguments)]
  pub fn price_call(
    &self,
    s1: f64,
    s2: f64,
    k: f64,
    r: f64,
    q1: f64,
    q2: f64,
    tau: f64,
  ) -> McEstimate<f64> {
    self.price_option(s1, s2, k, r, q1, q2, tau, OptionType::Call)
  }

  /// Price the spread put $\max(K-(S_1-S_2),0)$ at one query point.
  // One argument per term of the query.
  #[allow(clippy::too_many_arguments)]
  pub fn price_put(
    &self,
    s1: f64,
    s2: f64,
    k: f64,
    r: f64,
    q1: f64,
    q2: f64,
    tau: f64,
  ) -> McEstimate<f64> {
    self.price_option(s1, s2, k, r, q1, q2, tau, OptionType::Put)
  }
}

#[cfg(test)]
#[path = "spread_tests.rs"]
mod tests;
