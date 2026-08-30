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
use rayon::prelude::*;
use stochastic_rs_distributions::special::norm_cdf;

use crate::OptionType;
use crate::mc::McEstimate;
use crate::pricing::mc_stats::std_err_from_sums;
use crate::traits::FloatExt;

/// Floor a payoff at zero without letting the floor swallow a `NaN`.
///
/// `f64::max` returns the non-`NaN` operand, so `f64::NAN.max(0.0)` is
/// `0.0` — a floor and a poison check run together into one plausible
/// wrong answer. The floor itself is right: an exchange option and a
/// spread option both pay `max(·, 0)`, and a value a few ulp below zero
/// is round-off around a worthless contract. An undefined value has no
/// payoff to floor, so it travels on. Same split as
/// `pricing::fourier::pricer`'s `floor_price`.
#[inline]
fn floor_payoff(x: f64) -> f64 {
  if x.is_nan() { x } else { x.max(0.0) }
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
  /// Validating constructor.
  ///
  /// # Panics
  /// - if `sigma1` or `sigma2` is negative or `NaN` — not a volatility
  /// - if `rho` is outside `[-1, 1]` or `NaN` — not a correlation
  ///
  /// An out-of-range correlation is the sharp one here, because
  /// [`price`](Self::price)'s degenerate branch catches it and returns a
  /// number: at `rho = 5` the combined variance
  /// $\sigma_1^2+\sigma_2^2-2\rho\sigma_1\sigma_2$ goes *negative*, trips
  /// the `< 1e-14` test, and the exchange option prices as its discounted
  /// intrinsic — `10.0` against the correct `16.19`. At `rho = -5` the
  /// variance is merely too large and the price comes back `36.94`. Both
  /// volatilities are checked to the same standard and for the same
  /// reason as [`KirkSpreadPricer`](crate::pricing::kirk::KirkSpreadPricer)'s:
  /// validating one would swap the old asymmetry for a new one, and a
  /// negative `sigma1` prices at `21.21`.
  ///
  /// Admissible and still accepted: perfect correlation either way, and a
  /// zero-volatility leg. `sigma1 == sigma2` at `rho == 1` is exactly the
  /// degenerate branch, which is a limit rather than an error.
  ///
  /// No longer `const fn`. What made that safe was measured rather than
  /// assumed: **zero** `const` or `static` items of this type exist in the
  /// workspace, against 32 `MargrabePricer::new` call sites.
  pub fn new(sigma1: f64, sigma2: f64, rho: f64) -> Self {
    assert!(
      sigma1 >= 0.0,
      "MargrabePricer::new: sigma1 must be a non-negative volatility (got {sigma1})"
    );
    assert!(
      sigma2 >= 0.0,
      "MargrabePricer::new: sigma2 must be a non-negative volatility (got {sigma2})"
    );
    assert!(
      (-1.0..=1.0).contains(&rho),
      "MargrabePricer::new: rho must be in [-1, 1] (got {rho})"
    );
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

  /// Price the exchange option at one query point. This is always the call
  /// payoff $\max(S_1 - S_2, 0)$; the "put" version $\max(S_2 - S_1, 0)$ is
  /// the same model and the same query with the two legs swapped in both.
  ///
  /// The degenerate branch is the $\sigma \to 0$ limit — the spread is
  /// deterministic, so the option is worth its discounted intrinsic value
  /// — and it is reached by an *admissible* model, `sigma1 == sigma2` at
  /// `rho == 1`. It floors through `floor_payoff` rather than
  /// `f64::max`, because that branch is also where a `NaN` query lands: a
  /// `NaN` `tau` — which [`TimeExt::tau_or_from_dates`](crate::traits::TimeExt)
  /// returns for an expiry that never resolved — used to price a perfectly
  /// correlated exchange option at a confident `0.0`, while the same
  /// `tau` against any other model returns `NaN`.
  pub fn price(&self, s1: f64, s2: f64, q1: f64, q2: f64, tau: f64) -> f64 {
    let v_sq = self.combined_variance();
    if v_sq < 1e-14 {
      return floor_payoff(s1 * (-q1 * tau).exp() - s2 * (-q2 * tau).exp());
    }
    let (d1, d2) = Self::d1_d2(v_sq, s1, s2, q1, q2, tau);
    s1 * (-q1 * tau).exp() * norm_cdf(d1) - s2 * (-q2 * tau).exp() * norm_cdf(d2)
  }

  /// Greek delta with respect to $S_1$ at one query point.
  pub fn delta1(&self, s1: f64, s2: f64, q1: f64, q2: f64, tau: f64) -> f64 {
    let v_sq = self.combined_variance();
    if v_sq < 1e-14 {
      return (-q1 * tau).exp();
    }
    let (d1, _) = Self::d1_d2(v_sq, s1, s2, q1, q2, tau);
    (-q1 * tau).exp() * norm_cdf(d1)
  }

  /// Greek delta with respect to $S_2$ at one query point.
  pub fn delta2(&self, s1: f64, s2: f64, q1: f64, q2: f64, tau: f64) -> f64 {
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
pub struct McSpreadPricer {
  /// Volatility of asset 1.
  pub sigma1: f64,
  /// Volatility of asset 2.
  pub sigma2: f64,
  /// Correlation.
  pub rho: f64,
  /// Number of MC paths.
  pub n_paths: usize,
}

impl McSpreadPricer {
  /// Validating constructor.
  ///
  /// # Panics
  /// - if `sigma1` or `sigma2` is negative or `NaN` — not a volatility
  /// - if `rho` is outside `[-1, 1]` or `NaN` — not a correlation
  /// - if `n_paths` is `0`
  ///
  /// The correlation guard closes a second instance of the `f64::max`
  /// trap, one layer up from the payoff floor: the Cholesky-free factor
  /// `sqrt((1 - rho²).max(0.0))` *absorbs* an out-of-range correlation
  /// instead of announcing it, so the second asset is simulated as
  /// `rho·z1` alone and the spread call comes back `13.62` at `rho = 5`
  /// and `35.35` at `rho = -5`, against `10.69`. Both volatilities are
  /// checked to the same standard: a negative `sigma1` prices at `15.93`.
  ///
  /// `n_paths == 0` is the one guard here that does **not** close a wrong
  /// number — the empty average is `0/0` and the price is already `NaN`.
  /// It is rejected so that a path count is refused where it is supplied,
  /// matching
  /// [`GbmMalliavinPricer::new`](crate::pricing::malliavin_gbm::GbmMalliavinPricer::new),
  /// the crate's other Monte Carlo pricer holding its own path count.
  ///
  /// No longer `const fn`: **zero** `const` or `static` items of this type
  /// in the workspace, against 20 `McSpreadPricer::new` call sites.
  pub fn new(sigma1: f64, sigma2: f64, rho: f64, n_paths: usize) -> Self {
    assert!(
      sigma1 >= 0.0,
      "McSpreadPricer::new: sigma1 must be a non-negative volatility (got {sigma1})"
    );
    assert!(
      sigma2 >= 0.0,
      "McSpreadPricer::new: sigma2 must be a non-negative volatility (got {sigma2})"
    );
    assert!(
      (-1.0..=1.0).contains(&rho),
      "McSpreadPricer::new: rho must be in [-1, 1] (got {rho})"
    );
    assert!(
      n_paths >= 1,
      "McSpreadPricer::new: n_paths must be at least 1 (got {n_paths})"
    );
    Self {
      sigma1,
      sigma2,
      rho,
      n_paths,
    }
  }

  /// Price either leg at one query point with a single simulation.
  ///
  /// The per-path payoff floors through `floor_payoff`. With a bare
  /// `.max(0.0)` **every** poisoned path zeroed independently, so the
  /// average came back `0.0` rather than `NaN` — a whole simulation's
  /// worth of undefined payoffs reported as a worthless option. Both a
  /// `NaN` query coordinate (`s1`, `s2`, `k`, a dividend yield) and a
  /// `NaN` model parameter (`rho`, either volatility) reach it, the
  /// latter through the `pub` fields whatever [`new`](Self::new) accepts.
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
    <f64 as FloatExt>::fill_standard_normal_slice(&mut all_z);

    let payoff_of = |i: usize| -> f64 {
      let z1 = all_z[2 * i];
      let z2_indep = all_z[2 * i + 1];
      let z2 = rho * z1 + sqrt_one_minus_rho2 * z2_indep;
      let s1_t = s1 * (drift1 + vol1 * z1).exp();
      let s2_t = s2 * (drift2 + vol2 * z2).exp();
      floor_payoff(phi * (s1_t - s2_t - k))
    };
    let sum: f64 = (0..self.n_paths).into_par_iter().map(&payoff_of).sum();
    let sum_sq: f64 = (0..self.n_paths)
      .into_par_iter()
      .map(|i| {
        let y = payoff_of(i);
        y * y
      })
      .sum();

    let discount = (-r * tau).exp();
    McEstimate {
      mean: discount * sum / self.n_paths as f64,
      std_err: discount * std_err_from_sums(sum, sum_sq, self.n_paths),
      n_samples: self.n_paths,
    }
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
