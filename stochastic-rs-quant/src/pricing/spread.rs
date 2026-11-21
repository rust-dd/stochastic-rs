//! # Spread
//!
//! Margrabe (1978) exchange option ($K=0$):
//! $$
//! M = S_1 e^{-q_1 T} N(d_1) - S_2 e^{-q_2 T} N(d_2),\quad
//! \sigma = \sqrt{\sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2}
//! $$
//!
//! Source:
//! - Margrabe, W. (1978), "The Value of an Option to Exchange One Asset for Another", J. Finance 33
//! - Kirk, E. (1995), "Correlation in the Energy Markets" — see [`super::kirk`] for non-zero strikes.
//!
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_stochastic::mc::McEstimate;

use crate::OptionType;
use crate::pricing::digital::expired;
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

/// Margrabe (1978) exchange option: pays $\max(S_1 - S_2, 0)$.
///
/// $$
/// V = S_1 e^{-q_1 T} N(d_1) - S_2 e^{-q_2 T} N(d_2),\quad
/// \sigma = \sqrt{\sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2}
/// $$
///
/// The struct holds **model state only** — the two volatilities and their
/// correlation. The two spots, the two dividend yields and the maturity are
/// the pricing *query* and travel as arguments to [`price`](Self::price),
/// so one instance prices a whole spot/maturity grid.
///
/// The combined volatility $\sigma$ is the one place this differs from
/// [`KirkSpreadPricer`](crate::pricing::kirk::KirkSpreadPricer). Kirk
/// weights $\sigma_2$ by the query's own $F_2/(F_2+X)$ and therefore
/// *cannot* cache it; Margrabe's depends on the model alone and could be
/// cached, but is recomputed per call anyway, so no field on the struct is
/// ever a number left over from an earlier query.
///
/// There is no rate. An exchange option's two discount factors cancel, so
/// $r$ never enters the formula — an absence, not an omission.
///
/// A two-asset payoff with no strike carries no
/// [`ModelPricer`](crate::traits::ModelPricer), whose
/// `price_call(s, k, r, q, tau)` prices one underlying against one strike.
/// Margrabe belongs to the multi-asset "convention, no trait" family
/// alongside [`KirkSpreadPricer`](crate::pricing::kirk::KirkSpreadPricer)
/// and [`McSpreadPricer`], and exposes the same model/query split through
/// inherent methods.
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
  /// `tau = 0`, NaN past expiry.
  pub fn price(&self, s1: f64, s2: f64, q1: f64, q2: f64, tau: f64) -> f64 {
    if let Some(value) = expired(tau, &[q1, q2], || vanilla_payoff(OptionType::Call, s1, s2)) {
      return value;
    }
    let v_sq = self.combined_variance();
    if v_sq < 1e-14 {
      return (s1 * (-q1 * tau).exp() - s2 * (-q2 * tau).exp()).max_or_nan(0.0);
    }
    let (d1, d2) = Self::d1_d2(v_sq, s1, s2, q1, q2, tau);
    s1 * (-q1 * tau).exp() * norm_cdf(d1) - s2 * (-q2 * tau).exp() * norm_cdf(d2)
  }

  /// Delta with respect to $S_1$; NaN at and past expiry.
  pub fn delta1(&self, s1: f64, s2: f64, q1: f64, q2: f64, tau: f64) -> f64 {
    if tau.is_nan() || tau <= 0.0 {
      return f64::NAN;
    }
    let v_sq = self.combined_variance();
    if v_sq < 1e-14 {
      return (-q1 * tau).exp();
    }
    let (d1, _) = Self::d1_d2(v_sq, s1, s2, q1, q2, tau);
    (-q1 * tau).exp() * norm_cdf(d1)
  }

  /// Delta with respect to $S_2$; NaN at and past expiry.
  pub fn delta2(&self, s1: f64, s2: f64, q1: f64, q2: f64, tau: f64) -> f64 {
    if tau.is_nan() || tau <= 0.0 {
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

/// Monte-Carlo spread option pricer for general non-zero strikes under
/// correlated geometric Brownian motion. Pays
/// $\max\!\big(\phi(S_1 - S_2 - K), 0\big)$ where $\phi=\pm 1$.
///
/// The struct holds **model and method state only** — the two volatilities,
/// their correlation, and the Monte Carlo path count. The two spots, the
/// strike, the rate, the two dividend yields and the maturity are the
/// pricing *query* and travel as arguments, so one instance prices a whole
/// strike/maturity grid.
///
/// `n_paths` is neither model nor query but a convergence control, and it
/// sits beside the model for the same reason
/// `GbmMalliavinPricer` keeps its own path and step counts there: it fixes
/// how accurately this instance answers, not what it is answering about.
///
/// [`price_call`](Self::price_call) and [`price_put`](Self::price_put) each
/// run **one** simulation. Neither is derived from the other by put-call
/// parity, which would quote the two legs off different samples.
///
/// Like [`MargrabePricer`] this is a two-asset payoff and so carries no
/// [`ModelPricer`](crate::traits::ModelPricer); it follows the same
/// "convention, no trait" split through inherent methods.
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
