//! # Quanto option
//!
//! $$
//! c = E_p\left[S e^{(b - r)\tau} N(d_1) - K e^{-r\tau} N(d_2)\right],\qquad b = r_f - q - \rho\,\sigma_S\sigma_E
//! $$
//!
//! Fixed-exchange-rate foreign equity option (Reiner 1992): the payoff on a
//! foreign asset `S` is converted to the domestic currency at the fixed rate
//! `E_p`. Under the domestic risk-neutral measure the foreign asset drifts at
//! `r_f − q − ρ σ_S σ_E` — the quanto adjustment — so the price is `E_p` times
//! a Black–Scholes–Merton price with that cost of carry, discounted at the
//! domestic rate `r`, with `d₁ = [ln(S/K) + (b + σ_S²/2)τ] / (σ_S √τ)` and
//! `d₂ = d₁ − σ_S √τ`.
//!
//! References: Reiner, E. (1992), *Quanto mechanics*, Risk 5(3), 59–63;
//! Haug, E. G. (2007), *The Complete Guide to Option Pricing Formulas*, 2nd
//! ed., McGraw-Hill, §2.13.4.

use stochastic_rs_distributions::special::norm_cdf;

use crate::traits::ModelPricer;

/// Quanto (fixed exchange rate foreign equity) option pricer.
///
/// The struct holds **model state only**: the two volatilities, their
/// correlation, the foreign rate and the fixed conversion rate. The query
/// `(s, k, r, q, tau)` travels as arguments, with `r` the **domestic**
/// risk-free rate and `q` the asset's dividend yield.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantoPricer {
  /// Volatility σ_S of the foreign asset in its own currency.
  pub v: f64,
  /// Volatility σ_E of the exchange rate (domestic units per foreign unit).
  pub v_fx: f64,
  /// Correlation ρ between the asset and the exchange rate.
  pub rho: f64,
  /// Foreign risk-free rate r_f.
  pub r_f: f64,
  /// Fixed exchange rate E_p converting the foreign payoff into domestic
  /// currency.
  pub fixed_rate: f64,
}

impl QuantoPricer {
  pub const fn new(v: f64, v_fx: f64, rho: f64, r_f: f64, fixed_rate: f64) -> Self {
    Self {
      v,
      v_fx,
      rho,
      r_f,
      fixed_rate,
    }
  }

  /// Cost of carry of the foreign asset under the domestic measure,
  /// `b = r_f − q − ρ σ_S σ_E`.
  pub fn carry(&self, q: f64) -> f64 {
    self.r_f - q - self.rho * self.v * self.v_fx
  }

  /// Quanto forward `E_p · S · e^{bτ}`: the domestic-currency forward of
  /// the converted asset.
  pub fn forward(&self, s: f64, q: f64, tau: f64) -> f64 {
    self.fixed_rate * s * (self.carry(q) * tau).exp()
  }

  /// Call and put in domestic currency at the query `(s, k, r, q, tau)`.
  pub fn call_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> (f64, f64) {
    let b = self.carry(q);
    let sqrt_tau = tau.sqrt();
    let d1 = ((s / k).ln() + (b + 0.5 * self.v * self.v) * tau) / (self.v * sqrt_tau);
    let d2 = d1 - self.v * sqrt_tau;
    let carry = ((b - r) * tau).exp();
    let disc = (-r * tau).exp();
    let call = self.fixed_rate * (s * carry * norm_cdf(d1) - k * disc * norm_cdf(d2));
    let put = self.fixed_rate * (k * disc * norm_cdf(-d2) - s * carry * norm_cdf(-d1));
    (call, put)
  }
}

impl ModelPricer for QuantoPricer {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.call_put(s, k, r, q, tau).0
  }

  /// Overrides the trait's vanilla-parity default: the carry factor here is
  /// `e^{(b − r)τ}` with the quanto drift and the whole price is scaled by
  /// `E_p`, so parity reads `C − P = E_p (S e^{(b − r)τ} − K e^{−rτ})`.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.call_put(s, k, r, q, tau).1
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::OptionType;
  use crate::pricing::bsm::BSMCoc;
  use crate::pricing::bsm::BSMPricer;

  /// Haug (2007) §2.13.4 example inputs: `S = 100, K = 105, E_p = 1.5,
  /// τ = 0.5, r = 0.08, r_f = 0.05, q = 0.04, σ_S = 0.2, σ_E = 0.12, ρ = 0.3`.
  const MODEL: QuantoPricer = QuantoPricer::new(0.2, 0.12, 0.3, 0.05, 1.5);
  const Q: (f64, f64, f64, f64, f64) = (100.0, 105.0, 0.08, 0.04, 0.5);

  /// Reference: Reiner's formula evaluated with scipy 1.x:
  ///   from scipy.stats import norm; import numpy as np
  ///   b = 0.05 - 0.04 - 0.3 * 0.2 * 0.12
  ///   d1 = (np.log(100 / 105) + (b + 0.02) * 0.5) / (0.2 * np.sqrt(0.5)); d2 = d1 - 0.2 * np.sqrt(0.5)
  ///   1.5 * (100 * np.exp((b - 0.08) * 0.5) * norm.cdf(d1) - 105 * np.exp(-0.04) * norm.cdf(d2))    # 5.2936847941
  ///   1.5 * (105 * np.exp(-0.04) * norm.cdf(-d2) - 100 * np.exp((b - 0.08) * 0.5) * norm.cdf(-d1))  # 12.2976985036
  /// A 4·10⁶-path Monte Carlo of the domestic-measure joint lognormal
  /// (`S·E` drifting at `r − q`, `E` at `r − r_f`, no quanto-drift assumption)
  /// gave 5.29214 ± 0.00532 for the call. The tolerance follows the crate's
  /// `norm_cdf` (Abramowitz–Stegun 7.1.26 erf, 1.5e-7 absolute), which moves
  /// a price of this size by up to ~3e-5.
  #[test]
  fn matches_the_reiner_formula_on_the_haug_example() {
    let (s, k, r, q, tau) = Q;
    let (call, put) = MODEL.call_put(s, k, r, q, tau);
    assert!((call - 5.2936847941).abs() < 1e-4, "call {call}");
    assert!((put - 12.2976985036).abs() < 1e-4, "put {put}");
    assert!((MODEL.forward(s, q, tau) - 150.2101470686).abs() < 1e-9);
  }

  #[test]
  fn parity_carries_the_quanto_drift_and_the_fixed_rate() {
    let (s, k, r, q, tau) = Q;
    let (call, put) = MODEL.call_put(s, k, r, q, tau);
    let b = MODEL.carry(q);
    let want = 1.5 * (s * ((b - r) * tau).exp() - k * (-r * tau).exp());
    assert!(
      (call - put - want).abs() < 1e-9,
      "parity residual {}",
      call - put - want
    );
    assert_eq!(MODEL.price_call(s, k, r, q, tau), call);
    assert_eq!(MODEL.price_put(s, k, r, q, tau), put);
    assert_eq!(MODEL.price_option(s, k, r, q, tau, OptionType::Put), put);
  }

  /// With `ρ = 0` and `r_f = r` the quanto drift is the Merton carry
  /// `r − q`, so the price is exactly `E_p` times the Merton (1973) price.
  #[test]
  fn zero_correlation_and_equal_rates_reduce_to_the_scaled_merton_price() {
    let (s, k, r, q, tau) = Q;
    let quanto = QuantoPricer::new(0.2, 0.12, 0.0, r, 1.5);
    let merton = BSMPricer::new(0.2, BSMCoc::Merton1973);
    let (qc, qp) = quanto.call_put(s, k, r, q, tau);
    let (mc, mp) = merton.call_put(s, k, r, q, tau);
    assert!((qc - 1.5 * mc).abs() < 1e-12 && (qp - 1.5 * mp).abs() < 1e-12);
  }

  /// A positive asset–currency correlation lowers the quanto drift, hence
  /// the call, and raises the put.
  #[test]
  fn correlation_moves_the_prices_through_the_drift() {
    let (s, k, r, q, tau) = Q;
    let call = |rho: f64| QuantoPricer::new(0.2, 0.12, rho, 0.05, 1.5).call_put(s, k, r, q, tau);
    let (c_neg, p_neg) = call(-0.3);
    let (c_zero, p_zero) = call(0.0);
    let (c_pos, p_pos) = call(0.3);
    assert!(c_neg > c_zero && c_zero > c_pos);
    assert!(p_neg < p_zero && p_zero < p_pos);
  }
}
