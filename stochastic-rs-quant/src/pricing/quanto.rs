//! Fixed-exchange-rate foreign equity options (Reiner 1992): `E_p` times a Black–Scholes–Merton
//! price at the quanto carry `b`, discounted at the domestic rate.
//!
//! $$
//! c = E_p\left[S e^{(b - r)\tau} N(d_1) - K e^{-r\tau} N(d_2)\right],\qquad b = r_f - q - \rho\,\sigma_S\sigma_E
//! $$
//!
//! Reference: Reiner (1992), Risk 5(3), 59–63
//! Reference: Haug (2007), The Complete Guide to Option Pricing Formulas, 2nd ed., §2.13.4

use stochastic_rs_distributions::special::norm_cdf;

use crate::OptionType;
use crate::pricing::digital::expired;
use crate::pricing::digital::not_positive;
use crate::pricing::digital::vanilla_payoff;
use crate::pricing::exotic::assert_sigma;
use crate::traits::ModelPricer;

/// Quanto option pricer holding both volatilities, their correlation, the foreign rate and the
/// fixed rate `E_p`; in the query `r` is the domestic rate and `q` the asset's yield.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantoPricer {
  /// Volatility σ_S of the foreign asset in its own currency.
  pub sigma: f64,
  /// Volatility σ_E of the exchange rate (domestic units per foreign unit).
  pub sigma_fx: f64,
  /// Correlation ρ between the asset and the exchange rate.
  pub rho: f64,
  /// Foreign risk-free rate r_f.
  pub r_f: f64,
  /// Fixed exchange rate E_p converting the foreign payoff into domestic
  /// currency.
  pub fixed_rate: f64,
}

impl QuantoPricer {
  /// Panics unless both volatilities and `fixed_rate` are finite and `>= 0`, `rho ∈ [−1, 1]` and
  /// `r_f` finite; a `fixed_rate` of `0` converts the payoff to nothing and prices `0`.
  pub fn new(sigma: f64, sigma_fx: f64, rho: f64, r_f: f64, fixed_rate: f64) -> Self {
    assert_sigma(sigma);
    assert!(
      sigma_fx.is_finite() && sigma_fx >= 0.0,
      "sigma_fx must satisfy `sigma_fx.is_finite() && sigma_fx >= 0.0`, got sigma_fx = {sigma_fx:?}"
    );
    assert!(
      (-1.0..=1.0).contains(&rho),
      "rho must satisfy `(-1.0..=1.0).contains(&rho)`, got rho = {rho:?}"
    );
    assert!(
      r_f.is_finite(),
      "r_f must satisfy `r_f.is_finite()`, got r_f = {r_f:?}"
    );
    assert!(
      fixed_rate.is_finite() && fixed_rate >= 0.0,
      "fixed_rate must satisfy `fixed_rate.is_finite() && fixed_rate >= 0.0`, got fixed_rate = {fixed_rate:?}"
    );
    Self {
      sigma,
      sigma_fx,
      rho,
      r_f,
      fixed_rate,
    }
  }

  /// Cost of carry of the foreign asset under the domestic measure,
  /// `b = r_f − q − ρ σ_S σ_E`.
  pub fn carry(&self, q: f64) -> f64 {
    self.r_f - q - self.rho * self.sigma * self.sigma_fx
  }

  /// Quanto forward `E_p · S · e^{bτ}`: the domestic-currency forward of
  /// the converted asset.
  pub fn forward(&self, s: f64, q: f64, tau: f64) -> f64 {
    self.fixed_rate * s * (self.carry(q) * tau).exp()
  }

  /// `(S e^{(b − r)τ}, K e^{−rτ}, d1, d2)` at the query `(s, k, r, q, tau)`.
  fn legs(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> (f64, f64, f64, f64) {
    let b = self.carry(q);
    let sqrt_tau = tau.sqrt();
    let d1 = ((s / k).ln() + (b + 0.5 * self.sigma * self.sigma) * tau) / (self.sigma * sqrt_tau);
    let d2 = d1 - self.sigma * sqrt_tau;
    (s * ((b - r) * tau).exp(), k * (-r * tau).exp(), d1, d2)
  }

  /// `E_p` times the vanilla payoff at `tau = 0`; NaN past expiry and, before it, for a spot or
  /// strike that is not positive; `None` where the formula holds.
  fn off_formula(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, leg: OptionType) -> Option<f64> {
    let payoff = expired(tau, &[r, q], || self.fixed_rate * vanilla_payoff(leg, s, k));
    payoff.or_else(|| not_positive(&[s, k]).then_some(f64::NAN))
  }
}

impl ModelPricer for QuantoPricer {
  /// In domestic currency.
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if let Some(value) = self.off_formula(s, k, r, q, tau, OptionType::Call) {
      return value;
    }
    let (carried, strike, d1, d2) = self.legs(s, k, r, q, tau);
    self.fixed_rate * (carried * norm_cdf(d1) - strike * norm_cdf(d2))
  }

  /// Not the parity default: the carry is `e^{(b − r)τ}` with the quanto drift and the price is
  /// scaled by `E_p`, so parity reads `C − P = E_p (S e^{(b − r)τ} − K e^{−rτ})`.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if let Some(value) = self.off_formula(s, k, r, q, tau, OptionType::Put) {
      return value;
    }
    let (carried, strike, d1, d2) = self.legs(s, k, r, q, tau);
    self.fixed_rate * (strike * norm_cdf(-d2) - carried * norm_cdf(-d1))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::OptionType;
  use crate::pricing::bsm::BsmCoc;
  use crate::pricing::bsm::BsmPricer;

  /// Haug (2007) §2.13.4 example inputs: `S = 100, K = 105, E_p = 1.5,
  /// τ = 0.5, r = 0.08, r_f = 0.05, q = 0.04, σ_S = 0.2, σ_E = 0.12, ρ = 0.3`.
  fn model() -> QuantoPricer {
    QuantoPricer::new(0.2, 0.12, 0.3, 0.05, 1.5)
  }

  const Q: (f64, f64, f64, f64, f64) = (100.0, 105.0, 0.08, 0.04, 0.5);

  /// Reiner's formula by scipy 1.x at `b = 0.05 − 0.04 − 0.3·0.2·0.12`: call `5.2936847941`, put
  /// `12.2976985036`; a 4·10⁶-path joint lognormal MC, no quanto drift, gave `5.29214 ± 0.00532`.
  #[test]
  fn matches_the_reiner_formula_on_the_haug_example() {
    let (s, k, r, q, tau) = Q;
    let (call, put) = (
      model().price_call(s, k, r, q, tau),
      model().price_put(s, k, r, q, tau),
    );
    assert!((call - 5.2936847941).abs() < 1e-9, "call {call}");
    assert!((put - 12.2976985036).abs() < 1e-9, "put {put}");
    assert!((model().forward(s, q, tau) - 150.2101470686).abs() < 1e-9);
  }

  #[test]
  fn parity_carries_the_quanto_drift_and_the_fixed_rate() {
    let (s, k, r, q, tau) = Q;
    let (call, put) = (
      model().price_call(s, k, r, q, tau),
      model().price_put(s, k, r, q, tau),
    );
    let b = model().carry(q);
    let want = 1.5 * (s * ((b - r) * tau).exp() - k * (-r * tau).exp());
    assert!(
      (call - put - want).abs() < 1e-9,
      "parity residual {}",
      call - put - want
    );
    assert_eq!(model().price_call(s, k, r, q, tau), call);
    assert_eq!(model().price_put(s, k, r, q, tau), put);
    assert_eq!(model().price_option(s, k, r, q, tau, OptionType::Put), put);
  }

  /// With `ρ = 0` and `r_f = r` the quanto drift is the Merton carry
  /// `r − q`, so the price is exactly `E_p` times the Merton (1973) price.
  #[test]
  fn zero_correlation_and_equal_rates_reduce_to_the_scaled_merton_price() {
    let (s, k, r, q, tau) = Q;
    let quanto = QuantoPricer::new(0.2, 0.12, 0.0, r, 1.5);
    let merton = BsmPricer::new(0.2, BsmCoc::Merton1973);
    let (qc, qp) = (
      quanto.price_call(s, k, r, q, tau),
      quanto.price_put(s, k, r, q, tau),
    );
    let (mc, mp) = (
      merton.price_call(s, k, r, q, tau),
      merton.price_put(s, k, r, q, tau),
    );
    assert!((qc - 1.5 * mc).abs() < 1e-12 && (qp - 1.5 * mp).abs() < 1e-12);
  }

  /// A positive asset–currency correlation lowers the quanto drift, hence
  /// the call, and raises the put.
  #[test]
  fn correlation_moves_the_prices_through_the_drift() {
    let (s, k, r, q, tau) = Q;
    let legs = |rho: f64| {
      let model = QuantoPricer::new(0.2, 0.12, rho, 0.05, 1.5);
      (
        model.price_call(s, k, r, q, tau),
        model.price_put(s, k, r, q, tau),
      )
    };
    let (c_neg, p_neg) = legs(-0.3);
    let (c_zero, p_zero) = legs(0.0);
    let (c_pos, p_pos) = legs(0.3);
    assert!(c_neg > c_zero && c_zero > c_pos);
    assert!(p_neg < p_zero && p_zero < p_pos);
  }

  #[test]
  #[should_panic(
    expected = "sigma must satisfy `sigma.is_finite() && sigma >= 0.0`, got sigma = -0.2"
  )]
  fn new_rejects_a_negative_asset_volatility() {
    let _ = QuantoPricer::new(-0.2, 0.12, 0.3, 0.05, 1.5);
  }

  #[test]
  #[should_panic(
    expected = "sigma_fx must satisfy `sigma_fx.is_finite() && sigma_fx >= 0.0`, got sigma_fx = inf"
  )]
  fn new_rejects_an_infinite_exchange_rate_volatility() {
    let _ = QuantoPricer::new(0.2, f64::INFINITY, 0.3, 0.05, 1.5);
  }

  #[test]
  #[should_panic(expected = "rho must satisfy `(-1.0..=1.0).contains(&rho)`, got rho = 1.5")]
  fn new_rejects_a_correlation_past_one() {
    let _ = QuantoPricer::new(0.2, 0.12, 1.5, 0.05, 1.5);
  }

  #[test]
  #[should_panic(expected = "r_f must satisfy `r_f.is_finite()`, got r_f = NaN")]
  fn new_rejects_a_foreign_rate_that_is_not_finite() {
    let _ = QuantoPricer::new(0.2, 0.12, 0.3, f64::NAN, 1.5);
  }

  /// `E_p` converts the payoff at a price of domestic per foreign currency, which is never negative.
  #[test]
  #[should_panic(
    expected = "fixed_rate must satisfy `fixed_rate.is_finite() && fixed_rate >= 0.0`, got fixed_rate = -1.5"
  )]
  fn new_rejects_a_negative_fixed_rate() {
    let _ = QuantoPricer::new(0.2, 0.12, 0.3, 0.05, -1.5);
  }

  /// `E_p = 0` converts the payoff to nothing: a degenerate contract, worth `0` on both legs.
  #[test]
  fn a_zero_fixed_rate_prices_zero() {
    let model = QuantoPricer::new(0.2, 0.12, 0.3, 0.05, 0.0);
    let (s, k, r, q, tau) = Q;
    for tau in [tau, 0.0] {
      let legs = (
        model.price_call(s, k, r, q, tau),
        model.price_put(s, k, r, q, tau),
      );
      assert_eq!(legs, (0.0, 0.0), "{tau}");
    }
  }

  #[test]
  fn the_admissible_edges_stay_constructible() {
    for (sigma, sigma_fx, rho, r_f) in [(0.0, 0.0, -1.0, -0.01), (0.2, 0.12, 1.0, 0.0)] {
      let model = QuantoPricer::new(sigma, sigma_fx, rho, r_f, 1.5);
      assert_eq!(
        (model.sigma, model.sigma_fx, model.rho),
        (sigma, sigma_fx, rho)
      );
    }
  }
}
