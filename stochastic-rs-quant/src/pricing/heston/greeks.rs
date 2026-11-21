//! Heston (1993) Greeks by central differences: volatility Greeks chain `∂P/∂σ = 2√v0 ∂P/∂v0` from
//! the analytic `∂P/∂v0`; calendar Greeks are `−∂/∂τ`, against `AnalyticHestonEngine`'s `+∂P/∂τ`.

use super::HestonPricer;
use crate::OptionType;
use crate::traits::Greeks;
use crate::traits::ModelPricer;

impl HestonPricer {
  const H_TAU: f64 = 1e-5;
  const H_R: f64 = 1e-5;

  fn h_s(s: f64) -> f64 {
    s.abs() * 1e-4
  }

  fn h_v(&self) -> f64 {
    self.params.v0.abs().max(0.01) * 1e-4
  }

  /// Whether `σ = √v0` has a derivative here; `v0 == 0` has none, so the volatility-space Greeks
  /// return `NaN`.
  fn vol_chain_rule_is_defined(&self) -> bool {
    self.params.v0 > 0.0
  }

  /// Copy with `v0` bumped and floored at `1e-12`, as the analytic engine's down-bump is, so a bump
  /// near zero keeps the variance valid.
  fn with_v0_bump(&self, dv0: f64) -> Self {
    let mut bumped = *self;
    bumped.params.v0 = (bumped.params.v0 + dv0).max(1e-12);
    bumped
  }

  /// Delta — $\partial V/\partial S$.
  pub fn delta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    let h = Self::h_s(s);
    (self.price_option(s + h, k, r, q, tau, option_type)
      - self.price_option(s - h, k, r, q, tau, option_type))
      / (2.0 * h)
  }

  /// Gamma — $\partial^2 V/\partial S^2$. Option-type independent: the
  /// put differs from the call by terms linear in `S`.
  pub fn gamma(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    let h = Self::h_s(s);
    let p0 = self.price_call(s, k, r, q, tau);
    (self.price_call(s + h, k, r, q, tau) - 2.0 * p0 + self.price_call(s - h, k, r, q, tau))
      / (h * h)
  }

  /// Vega `∂V/∂σ` with `σ = √v0`; `NaN` at `v0 == 0`, where that chain rule is undefined.
  pub fn vega(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if !self.vol_chain_rule_is_defined() {
      return f64::NAN;
    }
    2.0 * self.params.v0.sqrt() * self.initial_variance_vega(s, k, r, q, tau)
  }

  /// Theta `∂V/∂t = −∂V/∂τ`; `NaN` for a non-finite `tau` or `tau ≤ 1e-5` (the step), where the
  /// down-bump would price past expiry.
  pub fn theta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    let h = Self::H_TAU;
    if !(tau.is_finite() && tau > h) {
      return f64::NAN;
    }
    -(self.price_option(s, k, r, q, tau + h, option_type)
      - self.price_option(s, k, r, q, tau - h, option_type))
      / (2.0 * h)
  }

  /// Rho — $\partial V/\partial r$.
  pub fn rho(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    let h = Self::H_R;
    (self.price_option(s, k, r + h, q, tau, option_type)
      - self.price_option(s, k, r - h, q, tau, option_type))
      / (2.0 * h)
  }

  /// Vanna `∂²V/∂S∂σ`; `NaN` at `v0 == 0`, where the `σ = √v0` chain rule is undefined.
  pub fn vanna(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if !self.vol_chain_rule_is_defined() {
      return f64::NAN;
    }
    let h = Self::h_s(s);
    let p_s_v0 = (self.initial_variance_vega(s + h, k, r, q, tau)
      - self.initial_variance_vega(s - h, k, r, q, tau))
      / (2.0 * h);
    2.0 * self.params.v0.sqrt() * p_s_v0
  }

  /// Charm `∂²V/∂S∂t` (delta decay); `NaN` for a non-finite `tau` or `tau ≤ 1e-5`, where the
  /// down-bump would price past expiry.
  pub fn charm(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    let ht = Self::H_TAU;
    if !(tau.is_finite() && tau > ht) {
      return f64::NAN;
    }
    let hs = Self::h_s(s);
    -(self.price_option(s + hs, k, r, q, tau + ht, option_type)
      - self.price_option(s + hs, k, r, q, tau - ht, option_type)
      - self.price_option(s - hs, k, r, q, tau + ht, option_type)
      + self.price_option(s - hs, k, r, q, tau - ht, option_type))
      / (4.0 * hs * ht)
  }

  /// Volga `∂²V/∂σ²`; `NaN` at `v0 == 0`, where the `σ = √v0` chain rule is undefined.
  pub fn volga(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if !self.vol_chain_rule_is_defined() {
      return f64::NAN;
    }
    let h = self.h_v();
    let p_v0v0 = (self.with_v0_bump(h).initial_variance_vega(s, k, r, q, tau)
      - self.with_v0_bump(-h).initial_variance_vega(s, k, r, q, tau))
      / (2.0 * h);
    4.0 * self.params.v0 * p_v0v0 + 2.0 * self.initial_variance_vega(s, k, r, q, tau)
  }

  /// Veta $\partial^2 V/\partial\sigma\,\partial t$: NaN at `v0 == 0`, where $\sigma=\sqrt{v_0}$
  /// has no chain rule, and for a `tau` that is not finite or not above the step `H_TAU`.
  pub fn veta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if !self.vol_chain_rule_is_defined() {
      return f64::NAN;
    }
    let h = Self::H_TAU;
    if !(tau.is_finite() && tau > h) {
      return f64::NAN;
    }
    let p_tau_v0 = (self.initial_variance_vega(s, k, r, q, tau + h)
      - self.initial_variance_vega(s, k, r, q, tau - h))
      / (2.0 * h);
    -2.0 * self.params.v0.sqrt() * p_tau_v0
  }

  /// Every Greek at one query point.
  pub fn greeks(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    option_type: OptionType,
  ) -> Greeks {
    if tau == 0.0 {
      return Greeks::nan();
    }
    Greeks {
      delta: self.delta(s, k, r, q, tau, option_type),
      gamma: self.gamma(s, k, r, q, tau),
      vega: self.vega(s, k, r, q, tau),
      theta: self.theta(s, k, r, q, tau, option_type),
      rho: self.rho(s, k, r, q, tau, option_type),
      vanna: self.vanna(s, k, r, q, tau),
      charm: self.charm(s, k, r, q, tau, option_type),
      volga: self.volga(s, k, r, q, tau),
      veta: self.veta(s, k, r, q, tau),
    }
  }
}
