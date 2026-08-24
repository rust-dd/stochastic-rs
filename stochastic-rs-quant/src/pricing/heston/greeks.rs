//! Central finite-difference Greeks for the Heston (1993) semi-closed-form
//! price, at an explicit `(s, k, r, q, tau)` query point.
//!
//! `vega`/`vanna`/`volga`/`veta` bump the variance parameter `v0` — not
//! `√v0` — mirroring
//! [`AnalyticHestonEngine::finite_diff_greeks`](crate::pricing::engines::AnalyticHestonEngine),
//! then convert to a volatility-space derivative via the chain rule
//! `σ = √v0`: `∂P/∂σ = 2√v0 · ∂P/∂v0` and its higher partials. The
//! `∂P/∂v0` building block itself is the analytic
//! [`HestonPricer::call_put_initial_variance_vega`] rather than a finite
//! difference, for precision — vanna/volga/veta then finite-difference
//! *that* analytic function instead of double finite-differencing the raw
//! price.
//!
//! `theta`/`charm`/`veta` use the calendar `-∂/∂τ` convention mandated by
//! [`GreeksExt::theta`](crate::traits::GreeksExt::theta)'s own doc
//! (`∂V/∂t`) and matching [`BSMPricer`](crate::pricing::bsm::BSMPricer)'s /
//! `Merton1976Pricer`'s Greeks — the negative of the raw `+∂P/∂τ` that
//! [`AnalyticHestonEngine::finite_diff_greeks`](crate::pricing::engines::AnalyticHestonEngine)
//! computes (that engine predates these and was never updated to match; see
//! `heston/tests.rs::heston_greeks_match_engine_bumps` for how the two are
//! reconciled in tests).
//!
//! **`NaN` is a deliberate return value here**, not an "unimplemented"
//! marker. `vega`/`vanna`/`volga`/`veta` divide through the `σ = √v0` chain
//! rule above, which is undefined at `v0 <= 0`; each guards that case
//! explicitly and returns `NaN` rather than a wrong finite number.
//! `theta`/`charm`/`veta` carry a second, independent guard on `tau`,
//! returning `NaN` when it is non-finite or not safely larger than the
//! central-difference step `H_TAU` — see
//! `heston_greeks_nan_at_degenerate_inputs` for both guards exercised
//! directly.

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
    self.v0.abs().max(0.01) * 1e-4
  }

  /// Copy with `v0` bumped, floored at `1e-12` (mirrors
  /// [`AnalyticHestonEngine`](crate::pricing::engines::AnalyticHestonEngine)'s
  /// own down-bump clamp) so a downward variance bump near zero cannot
  /// produce a negative, model-invalid variance.
  fn with_v0_bump(&self, dv0: f64) -> Self {
    let mut bumped = *self;
    bumped.v0 = (bumped.v0 + dv0).max(1e-12);
    bumped
  }

  /// `∂(price)/∂v0`, analytic (no finite difference) via
  /// [`call_put_initial_variance_vega`](HestonPricer::call_put_initial_variance_vega).
  /// Identical for call and put per that method's own doc, which is why the
  /// four Greeks built on it take no option type.
  fn v0_vega(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.call_put_initial_variance_vega(s, k, r, q, tau).0
  }

  /// Delta — $\partial V/\partial S$.
  pub fn delta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    let h = Self::h_s(s);
    (self.price_option(s + h, k, r, q, tau, option_type)
      - self.price_option(s - h, k, r, q, tau, option_type))
      / (2.0 * h)
  }

  /// Gamma — $\partial^2 V/\partial S^2$. Option-type independent: the
  /// put differs from the call by terms linear in `S`.
  pub fn gamma(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let h = Self::h_s(s);
    let p0 = self.price_call(s, k, r, q, tau);
    (self.price_call(s + h, k, r, q, tau) - 2.0 * p0 + self.price_call(s - h, k, r, q, tau))
      / (h * h)
  }

  /// Vega — $\partial V/\partial\sigma$ with $\sigma=\sqrt{v_0}$.
  pub fn vega(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if self.v0 <= 0.0 {
      return f64::NAN;
    }
    2.0 * self.v0.sqrt() * self.v0_vega(s, k, r, q, tau)
  }

  /// Theta — $\partial V/\partial t$ (calendar convention, $-\partial/\partial\tau$).
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
    let h = Self::H_R;
    (self.price_option(s, k, r + h, q, tau, option_type)
      - self.price_option(s, k, r - h, q, tau, option_type))
      / (2.0 * h)
  }

  /// Vanna — $\partial^2 V/\partial S\partial\sigma$.
  pub fn vanna(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if self.v0 <= 0.0 {
      return f64::NAN;
    }
    let h = Self::h_s(s);
    let p_s_v0 =
      (self.v0_vega(s + h, k, r, q, tau) - self.v0_vega(s - h, k, r, q, tau)) / (2.0 * h);
    2.0 * self.v0.sqrt() * p_s_v0
  }

  /// Charm — $\partial^2 V/\partial S\partial t$ (delta decay).
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

  /// Volga / vomma — $\partial^2 V/\partial\sigma^2$ (vega convexity).
  pub fn volga(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if self.v0 <= 0.0 {
      return f64::NAN;
    }
    let h = self.h_v();
    let p_v0v0 = (self.with_v0_bump(h).v0_vega(s, k, r, q, tau)
      - self.with_v0_bump(-h).v0_vega(s, k, r, q, tau))
      / (2.0 * h);
    4.0 * self.v0 * p_v0v0 + 2.0 * self.v0_vega(s, k, r, q, tau)
  }

  /// Veta — $\partial^2 V/\partial\sigma\partial t$ (vega decay).
  pub fn veta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if self.v0 <= 0.0 {
      return f64::NAN;
    }
    let h = Self::H_TAU;
    if !(tau.is_finite() && tau > h) {
      return f64::NAN;
    }
    let p_tau_v0 =
      (self.v0_vega(s, k, r, q, tau + h) - self.v0_vega(s, k, r, q, tau - h)) / (2.0 * h);
    -2.0 * self.v0.sqrt() * p_tau_v0
  }

  /// Every Greek at one query point, in a single [`Greeks`] struct.
  ///
  /// The `volga → Greeks::volga` and `veta → Greeks::veta` mapping lives
  /// here and nowhere else, so a caller cannot get it wrong by
  /// hand-writing the nine-field literal.
  pub fn greeks(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    option_type: OptionType,
  ) -> Greeks {
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
