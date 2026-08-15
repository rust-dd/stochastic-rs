//! Pricing traits — `PricerExt`, `ModelPricer`, `GreeksExt`.

use crate::OptionType;

pub trait PricerExt: super::time::TimeExt {
  fn calculate_call_put(&self) -> (f64, f64);

  fn calculate_price(&self) -> f64;

  fn implied_volatility(&self, _c_price: f64, _option_type: OptionType) -> f64 {
    f64::NAN
  }
}

/// Aggregate Greek values produced by [`GreeksExt::greeks`].
///
/// Members default to [`f64::NAN`] so consumers can identify Greeks the
/// pricer does not expose. First-order: [`delta`](Self::delta),
/// [`vega`](Self::vega), [`theta`](Self::theta), [`rho`](Self::rho).
/// Second-order: [`gamma`](Self::gamma), [`vanna`](Self::vanna),
/// [`charm`](Self::charm), [`volga`](Self::volga), [`veta`](Self::veta).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Greeks {
  pub delta: f64,
  pub gamma: f64,
  pub vega: f64,
  pub theta: f64,
  pub rho: f64,
  pub vanna: f64,
  pub charm: f64,
  pub volga: f64,
  pub veta: f64,
}

impl Default for Greeks {
  fn default() -> Self {
    Self::nan()
  }
}

impl Greeks {
  /// All-NaN sentinel — used as the Greeks of a pricer that exposes nothing.
  pub const fn nan() -> Self {
    Self {
      delta: f64::NAN,
      gamma: f64::NAN,
      vega: f64::NAN,
      theta: f64::NAN,
      rho: f64::NAN,
      vanna: f64::NAN,
      charm: f64::NAN,
      volga: f64::NAN,
      veta: f64::NAN,
    }
  }

  /// Component index order used by the [`as_array`](Self::as_array)
  /// accessor. Stable so downstream callers can hard-code positional
  /// access (`out[0] == delta` etc).
  pub const COMPONENT_NAMES: [&'static str; 9] = [
    "delta", "gamma", "vega", "theta", "rho", "vanna", "charm", "volga", "veta",
  ];

  /// Flatten into the canonical 9-element array matching
  /// [`COMPONENT_NAMES`](Self::COMPONENT_NAMES).
  pub fn as_array(&self) -> [f64; 9] {
    [
      self.delta, self.gamma, self.vega, self.theta, self.rho, self.vanna, self.charm, self.volga,
      self.veta,
    ]
  }
}

/// Trait for models that can price European options at arbitrary (K, T) points.
///
/// Unlike [`PricerExt`], which bundles market data and strike into the pricer,
/// `ModelPricer` separates the model from the pricing query. This enables
/// vectorized pricing across strike/maturity grids for calibration and vol
/// surface construction.
pub trait ModelPricer {
  /// Price a European call option.
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64;

  /// Price a European put via put-call parity.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let call = self.price_call(s, k, r, q, tau);
    call - s * (-q * tau).exp() + k * (-r * tau).exp()
  }

  /// Price a call or put.
  fn price_option(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    match option_type {
      OptionType::Call => self.price_call(s, k, r, q, tau),
      OptionType::Put => self.price_put(s, k, r, q, tau),
    }
  }
}

/// Common interface for Greeks reporting.
///
/// Pricers expose Greeks via inherent methods today (`BSMPricer::delta`,
/// `CashOrNothingPricer::delta`, …) — this trait gives generic / heterogeneous
/// code a single dispatch point. Only [`delta`](Self::delta) is required;
/// pricers that don't compute the higher-order Greeks return [`f64::NAN`]
/// from the default impls.
///
/// Pricers may have multiple Greek variants (analytical, Malliavin, finite
/// difference) — the trait exposes the canonical form. For Malliavin /
/// pathwise Greeks call the inherent methods (`malliavin_greeks::*::delta`)
/// directly.
///
/// First-order: [`delta`](Self::delta), [`vega`](Self::vega),
/// [`theta`](Self::theta), [`rho`](Self::rho).
/// Second-order: [`gamma`](Self::gamma), [`vanna`](Self::vanna),
/// [`charm`](Self::charm), [`volga`](Self::volga), [`veta`](Self::veta).
pub trait GreeksExt {
  /// Delta — $\partial V / \partial S$.
  fn delta(&self) -> f64;

  /// Gamma — $\partial^2 V / \partial S^2$. Defaults to NaN when not implemented.
  fn gamma(&self) -> f64 {
    f64::NAN
  }

  /// Vega — $\partial V / \partial \sigma$. Defaults to NaN when not implemented.
  fn vega(&self) -> f64 {
    f64::NAN
  }

  /// Theta — $\partial V / \partial t$. Defaults to NaN when not implemented.
  fn theta(&self) -> f64 {
    f64::NAN
  }

  /// Rho — $\partial V / \partial r$. Defaults to NaN when not implemented.
  fn rho(&self) -> f64 {
    f64::NAN
  }

  /// Vanna — $\partial^2 V / \partial S \partial \sigma$ (DvegaDspot).
  /// Defaults to NaN when not implemented.
  fn vanna(&self) -> f64 {
    f64::NAN
  }

  /// Charm — $\partial^2 V / \partial S \partial t$ (delta decay).
  /// Defaults to NaN when not implemented.
  fn charm(&self) -> f64 {
    f64::NAN
  }

  /// Volga / vomma — $\partial^2 V / \partial \sigma^2$ (vega convexity).
  /// Defaults to NaN when not implemented.
  fn volga(&self) -> f64 {
    f64::NAN
  }

  /// Veta — $\partial^2 V / \partial \sigma \partial t$ (vega decay).
  /// Defaults to NaN when not implemented.
  fn veta(&self) -> f64 {
    f64::NAN
  }

  /// Aggregate every Greek into a single [`Greeks`] struct.
  ///
  /// The default impl simply calls every accessor — fine for analytical
  /// pricers where each method is deterministic. **Monte Carlo pricers
  /// must override this method**, because calling each Greek individually
  /// would run a fresh independent simulation and produce a [`Greeks`]
  /// struct that mixes estimators from disjoint sample paths
  /// (mathematically inconsistent — e.g. delta/gamma sourced from different
  /// random draws). MC pricers should compute every Greek that can share
  /// paths in a single pass; see [`crate::pricing::malliavin_greeks`] for
  /// a worked example.
  fn greeks(&self) -> Greeks {
    Greeks {
      delta: self.delta(),
      gamma: self.gamma(),
      vega: self.vega(),
      theta: self.theta(),
      rho: self.rho(),
      vanna: self.vanna(),
      charm: self.charm(),
      volga: self.volga(),
      veta: self.veta(),
    }
  }
}

#[cfg(test)]
mod greeks_array_tests {
  use super::Greeks;

  #[test]
  fn as_array_matches_component_names_order() {
    let g = Greeks {
      delta: 0.5,
      gamma: 0.1,
      vega: 0.2,
      theta: -0.05,
      rho: 0.3,
      vanna: 0.4,
      charm: 0.05,
      volga: 0.6,
      veta: -0.02,
    };
    let arr = g.as_array();
    assert_eq!(arr.len(), Greeks::COMPONENT_NAMES.len());
    assert_eq!(arr[0], g.delta);
    assert_eq!(arr[8], g.veta);
  }
}
