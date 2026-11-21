//! Pricing traits — `ModelPricer`, `GreeksExt`.

use stochastic_rs_stochastic::mc::McEstimate;

use crate::OptionType;

/// Returned by the pricers' `greeks(…)` and [`GreeksExt::greeks`]: NaN for a Greek a pricer does
/// not expose, and for every Greek at `tau = 0`, where an expired option has a value only.
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

  /// Component order of [`as_array`](Self::as_array), stable for positional access (`out[0]` is
  /// delta).
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

/// Prices a call or put at any `(s, k, r, q, tau)` query with the model on the struct; exercise is
/// the implementor's. A bad parameter panics, an off-domain query is `NaN`, never a sentinel `0.0`.
pub trait ModelPricer {
  /// Price a call option.
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64;

  /// Put by European parity; override unless the carry is `e^{−qτ}` and exercise European, as the
  /// default's finite, wrong put is undetectable (an override may be `NaN`, as `SuperSharePricer`).
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

/// Marks `price_call` as a European vanilla `(S_T − K)⁺` on [`Self::vanilla_call_forward`], which
/// `ModelSurface`'s Black inversion needs: any other payoff inverts to a finite, wrong vol.
pub trait VanillaEuropeanCall: ModelPricer {
  /// Forward `price_call` is struck on, `Se^{(r−q)τ}` unless the carry differs (`Black1976`: `S`);
  /// `NaN` where this instance is no European vanilla (American `FiniteDifferencePricer`).
  fn vanilla_call_forward(&self, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    s * ((r - q) * tau).exp()
  }
}

/// No-argument Greeks of the Monte Carlo Malliavin estimators, one pass for all (pricers use their
/// inherent `greeks(…)`); only `delta` is required, an uncomputed one is `None`, `NaN` in `greeks`.
pub trait GreeksExt {
  /// Delta — $\partial V / \partial S$.
  fn delta(&self) -> Option<McEstimate<f64>>;

  /// Gamma — $\partial^2 V / \partial S^2$; `None` when the estimator does not expose it.
  fn gamma(&self) -> Option<McEstimate<f64>> {
    None
  }

  /// Vega — $\partial V / \partial \sigma$; `None` when not exposed.
  fn vega(&self) -> Option<McEstimate<f64>> {
    None
  }

  /// Theta — $\partial V / \partial t$; `None` when not exposed.
  fn theta(&self) -> Option<McEstimate<f64>> {
    None
  }

  /// Rho — $\partial V / \partial r$; `None` when not exposed.
  fn rho(&self) -> Option<McEstimate<f64>> {
    None
  }

  /// Vanna — $\partial^2 V / \partial S \partial \sigma$; `None` when not exposed.
  fn vanna(&self) -> Option<McEstimate<f64>> {
    None
  }

  /// Charm — $\partial^2 V / \partial S \partial t$; `None` when not exposed.
  fn charm(&self) -> Option<McEstimate<f64>> {
    None
  }

  /// Volga — $\partial^2 V / \partial \sigma^2$; `None` when not exposed.
  fn volga(&self) -> Option<McEstimate<f64>> {
    None
  }

  /// Veta — $\partial^2 V / \partial \sigma \partial t$; `None` when not exposed.
  fn veta(&self) -> Option<McEstimate<f64>> {
    None
  }

  /// Every Greek's mean in one [`Greeks`], NaN where an accessor is `None`. A Monte Carlo estimator
  /// overrides this to draw all of them from one simulation instead of one per accessor.
  fn greeks(&self) -> Greeks {
    Greeks {
      delta: mean_or_nan(self.delta()),
      gamma: mean_or_nan(self.gamma()),
      vega: mean_or_nan(self.vega()),
      theta: mean_or_nan(self.theta()),
      rho: mean_or_nan(self.rho()),
      vanna: mean_or_nan(self.vanna()),
      charm: mean_or_nan(self.charm()),
      volga: mean_or_nan(self.volga()),
      veta: mean_or_nan(self.veta()),
    }
  }
}

fn mean_or_nan(estimate: Option<McEstimate<f64>>) -> f64 {
  estimate.map_or(f64::NAN, |e| e.mean)
}

#[cfg(test)]
mod greeks_array_tests {
  use stochastic_rs_stochastic::mc::McEstimate;

  use super::Greeks;
  use super::GreeksExt;

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

  struct DeltaOnly;

  impl GreeksExt for DeltaOnly {
    fn delta(&self) -> Option<McEstimate<f64>> {
      Some(McEstimate {
        mean: 0.5,
        std_err: 0.01,
        n_samples: 100,
      })
    }
  }

  #[test]
  fn an_unexposed_greek_is_none_and_nan_in_the_aggregate() {
    let g = DeltaOnly;
    assert_eq!(g.delta().map(|e| e.mean), Some(0.5));
    assert_eq!(g.vega(), None);
    let all = g.greeks();
    assert_eq!(all.delta, 0.5);
    assert!(all.vega.is_nan());
  }
}
