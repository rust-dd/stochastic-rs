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
  /// Builds the model from the two volatilities and their correlation.
  pub const fn new(sigma1: f64, sigma2: f64, rho: f64) -> Self {
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
/// assert!(atm > 0.0);
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
  /// Builds the model from the two volatilities and their correlation,
  /// plus the Monte Carlo path count every price off this instance uses.
  pub const fn new(sigma1: f64, sigma2: f64, rho: f64, n_paths: usize) -> Self {
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
  ) -> f64 {
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

    let sum: f64 = (0..self.n_paths)
      .into_par_iter()
      .map(|i| {
        let z1 = all_z[2 * i];
        let z2_indep = all_z[2 * i + 1];
        let z2 = rho * z1 + sqrt_one_minus_rho2 * z2_indep;
        let s1_t = s1 * (drift1 + vol1 * z1).exp();
        let s2_t = s2 * (drift2 + vol2 * z2).exp();
        floor_payoff(phi * (s1_t - s2_t - k))
      })
      .sum();

    (-r * tau).exp() * sum / self.n_paths as f64
  }

  /// Price the spread call $\max(S_1-S_2-K,0)$ at one query point.
  pub fn price_call(&self, s1: f64, s2: f64, k: f64, r: f64, q1: f64, q2: f64, tau: f64) -> f64 {
    self.price_option(s1, s2, k, r, q1, q2, tau, OptionType::Call)
  }

  /// Price the spread put $\max(K-(S_1-S_2),0)$ at one query point.
  pub fn price_put(&self, s1: f64, s2: f64, k: f64, r: f64, q1: f64, q2: f64, tau: f64) -> f64 {
    self.price_option(s1, s2, k, r, q1, q2, tau, OptionType::Put)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Cross-arch tolerance: these goldens come from `norm_cdf`, whose last
  /// bit is a hostage to FMA contraction and libm differences between the
  /// aarch64-darwin dev machine and CI's ubuntu x86_64.
  const TOL: f64 = 1e-12;

  /// Values captured from the bundled-market-data `MargrabePricer` **before**
  /// the model/query reshape. The reshape is an API change only, so these
  /// must not move.
  #[test]
  fn margrabe_matches_pre_refactor_goldens() {
    let atm = MargrabePricer::new(0.20, 0.20, 0.0);
    let price = atm.price(100.0, 100.0, 0.0, 0.0, 1.0);
    assert!((price - 11.246296562219548).abs() < TOL, "atm {price}");
    let d1 = atm.delta1(100.0, 100.0, 0.0, 0.0, 1.0);
    assert!((d1 - 0.5562314828110977).abs() < TOL, "delta1 {d1}");
    let d2 = atm.delta2(100.0, 100.0, 0.0, 0.0, 1.0);
    assert!((d2 + 0.44376851718890226).abs() < TOL, "delta2 {d2}");

    let itm = MargrabePricer::new(0.20, 0.20, 0.5);
    let price = itm.price(200.0, 100.0, 0.01, 0.02, 0.5);
    assert!((price - 99.99751393839698).abs() < TOL, "itm {price}");

    let skewed = MargrabePricer::new(0.31, 0.17, -0.25);
    let price = skewed.price(95.0, 105.0, 0.03, 0.011, 2.25);
    assert!((price - 15.76555742239379).abs() < TOL, "skewed {price}");
    let d1 = skewed.delta1(95.0, 105.0, 0.03, 0.011, 2.25);
    assert!((d1 - 0.4848890956486614).abs() < TOL, "delta1 {d1}");
    let d2 = skewed.delta2(95.0, 105.0, 0.03, 0.011, 2.25);
    assert!((d2 + 0.28856101584980043).abs() < TOL, "delta2 {d2}");
  }

  /// One model instance prices a whole query grid — the point of the split.
  #[test]
  fn margrabe_one_model_prices_a_spot_grid() {
    let model = MargrabePricer::new(0.25, 0.20, 0.4);
    let prices = [90.0, 100.0, 110.0].map(|s1| model.price(s1, 100.0, 0.0, 0.0, 1.0));
    assert!(
      prices[0] < prices[1] && prices[1] < prices[2],
      "the exchange option must rise in S1: {prices:?}"
    );
  }

  /// The maturity is a query argument, so one instance covers a term
  /// structure. A stale `tau` cached at construction would return the same
  /// number three times.
  #[test]
  fn margrabe_one_model_prices_a_maturity_grid() {
    let model = MargrabePricer::new(0.25, 0.20, 0.4);
    let prices = [0.25, 1.0, 4.0].map(|tau| model.price(100.0, 100.0, 0.0, 0.0, tau));
    assert!(
      prices[0] < prices[1] && prices[1] < prices[2],
      "an at-the-money exchange option must rise in tau: {prices:?}"
    );
  }

  /// Margrabe with σ1=σ2 and ρ=1 must equal $\max(S_1 e^{-q_1 T} - S_2
  /// e^{-q_2 T}, 0)$ — the spread is deterministic at maturity.
  #[test]
  fn margrabe_perfect_correlation_equal_vol() {
    let price = MargrabePricer::new(0.2, 0.2, 1.0).price(100.0, 100.0, 0.0, 0.0, 1.0);
    assert!(price.abs() < 1e-8, "perfect-corr Margrabe={price}");
  }

  /// Margrabe at-the-money with zero correlation, equal vols.
  /// $S_1 = S_2 = 100$, $\sigma_1 = \sigma_2 = 0.20$, $\rho = 0$, $T = 1$
  /// → $\sigma_M = \sqrt{0.08} \approx 0.2828$
  /// → V = 100 * (2N(σ_M/2) - 1) ≈ 11.246
  #[test]
  fn margrabe_atm_zero_corr() {
    let price = MargrabePricer::new(0.20, 0.20, 0.0).price(100.0, 100.0, 0.0, 0.0, 1.0);
    let expected = 11.246;
    assert!((price - expected).abs() < 0.05, "Margrabe ATM={price}");
  }

  /// Margrabe with $S_1 \gg S_2$ approaches the discounted intrinsic.
  #[test]
  fn margrabe_deep_itm() {
    let price = MargrabePricer::new(0.20, 0.20, 0.5).price(200.0, 100.0, 0.01, 0.02, 0.5);
    let intrinsic = 200.0 * (-0.01_f64 * 0.5).exp() - 100.0 * (-0.02_f64 * 0.5).exp();
    assert!(
      price > intrinsic,
      "Margrabe deep ITM={price} vs intrinsic={intrinsic}"
    );
  }

  /// One Monte Carlo model instance prices a whole strike grid, both legs.
  /// The strikes are far enough apart that the ordering survives the
  /// sampling error of independent simulations.
  #[test]
  fn mc_spread_one_model_prices_a_strike_grid() {
    let model = McSpreadPricer::new(0.30, 0.25, 0.3, 100_000);
    let calls = [0.0, 10.0, 25.0].map(|k| model.price_call(110.0, 100.0, k, 0.03, 0.0, 0.0, 1.0));
    let puts = [0.0, 10.0, 25.0].map(|k| model.price_put(110.0, 100.0, k, 0.03, 0.0, 0.0, 1.0));
    assert!(
      calls[0] > calls[1] && calls[1] > calls[2],
      "spread calls must decay in the strike: {calls:?}"
    );
    assert!(
      puts[0] < puts[1] && puts[1] < puts[2],
      "spread puts must rise in the strike: {puts:?}"
    );
  }

  /// A `NaN` maturity on the degenerate-volatility branch used to price a
  /// confident **`0.0`**.
  ///
  /// The branch is reached by an admissible model — `sigma1 == sigma2` at
  /// `rho == 1`, whose combined variance is exactly zero — and `tau`
  /// arrives as `NaN` legitimately, from
  /// [`TimeExt::tau_or_from_dates`](crate::traits::TimeExt) on an expiry
  /// that never resolved. The second half is what made it a defect rather
  /// than a quirk: the *same* `NaN` `tau` against a non-degenerate model
  /// returns `NaN`, so one exchange option in a book reported no value
  /// while its neighbour reported no answer.
  #[test]
  fn margrabe_does_not_launder_a_nan_query_on_the_degenerate_branch() {
    let degenerate = MargrabePricer::new(0.2, 0.2, 1.0);
    assert_eq!(
      degenerate.combined_variance(),
      0.0,
      "this model must actually reach the degenerate branch"
    );
    for (name, got) in [
      ("tau", degenerate.price(100.0, 100.0, 0.0, 0.0, f64::NAN)),
      ("s1", degenerate.price(f64::NAN, 100.0, 0.0, 0.0, 1.0)),
      ("q1", degenerate.price(100.0, 100.0, f64::NAN, 0.0, 1.0)),
    ] {
      assert!(got.is_nan(), "a NaN {name} must not price: got {got}");
    }
    // The non-degenerate model already propagated, and must keep doing so.
    assert!(
      MargrabePricer::new(0.25, 0.20, 0.4)
        .price(100.0, 100.0, 0.0, 0.0, f64::NAN)
        .is_nan()
    );
    // The floor itself is untouched: the branch is still the discounted
    // intrinsic, floored at zero.
    assert_eq!(degenerate.price(100.0, 120.0, 0.0, 0.0, 1.0), 0.0);
    assert!((degenerate.price(120.0, 100.0, 0.0, 0.0, 1.0) - 20.0).abs() < 1e-12);
  }

  /// The per-path `max(0)` floor zeroed **every** poisoned payoff
  /// independently, so the average of a fully undefined simulation came
  /// back as `0.0` rather than `NaN`.
  ///
  /// Both routes are pinned. A `NaN` query coordinate is the ordinary one.
  /// A `NaN` *model* `rho` is written straight to the field rather than
  /// passed to the constructor: the fields are `pub`, so the estimator is
  /// reachable in that state whatever `new` chooses to accept.
  #[test]
  fn mc_spread_does_not_launder_a_nan_into_a_zero_price() {
    let model = McSpreadPricer::new(0.25, 0.20, 0.4, 2_000);
    for (name, got) in [
      (
        "s1",
        model.price_call(f64::NAN, 100.0, 10.0, 0.02, 0.0, 0.0, 1.0),
      ),
      (
        "s2",
        model.price_call(110.0, f64::NAN, 10.0, 0.02, 0.0, 0.0, 1.0),
      ),
      (
        "k",
        model.price_call(110.0, 100.0, f64::NAN, 0.02, 0.0, 0.0, 1.0),
      ),
      (
        "q1",
        model.price_call(110.0, 100.0, 10.0, 0.02, f64::NAN, 0.0, 1.0),
      ),
      (
        "put s1",
        model.price_put(f64::NAN, 100.0, 10.0, 0.02, 0.0, 0.0, 1.0),
      ),
    ] {
      assert!(got.is_nan(), "a NaN {name} must not price: got {got}");
    }

    let mut poisoned = McSpreadPricer::new(0.25, 0.20, 0.4, 2_000);
    poisoned.rho = f64::NAN;
    let got = poisoned.price_call(110.0, 100.0, 10.0, 0.02, 0.0, 0.0, 1.0);
    assert!(got.is_nan(), "a NaN model rho must not price: got {got}");

    poisoned = McSpreadPricer::new(0.25, 0.20, 0.4, 2_000);
    poisoned.sigma1 = f64::NAN;
    let got = poisoned.price_call(110.0, 100.0, 10.0, 0.02, 0.0, 0.0, 1.0);
    assert!(got.is_nan(), "a NaN model sigma1 must not price: got {got}");

    // The floor is still a floor: a deep out-of-the-money spread call is
    // worth zero, not a small negative number.
    let deep = model.price_call(110.0, 100.0, 500.0, 0.02, 0.0, 0.0, 1.0);
    assert_eq!(deep, 0.0, "the max(0) floor must survive: {deep}");
  }

  /// Margrabe ↔ MC (K=0) consistency: with enough paths the MC spread call
  /// should match Margrabe within 1.5%.
  #[test]
  fn margrabe_matches_mc_zero_strike() {
    let m_price = MargrabePricer::new(0.25, 0.20, 0.4).price(110.0, 100.0, 0.0, 0.0, 1.0);
    let mc = McSpreadPricer::new(0.25, 0.20, 0.4, 100_000);
    let mc_price = mc.price_call(110.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let rel = (m_price - mc_price).abs() / m_price;
    assert!(rel < 0.02, "margrabe={m_price}, mc={mc_price}, rel={rel}");
  }
}
