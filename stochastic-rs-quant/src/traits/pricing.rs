//! Pricing traits — `ModelPricer`, `GreeksExt`.

use crate::OptionType;

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

/// Trait for models that can price a single-underlying call or put at
/// arbitrary (K, T) points.
///
/// The struct holds model parameters; the query travels as arguments. That
/// separation is what enables vectorized pricing across strike/maturity
/// grids for calibration and vol surface construction — the retired
/// `PricerExt` bundled market data and strike into the pricer instead, so
/// a second query point meant a second pricer.
///
/// Exercise style is the **implementor's** choice, not the trait's: most
/// members price European exercise, but the American approximations
/// (`BjerksundStensland2002Pricer`, `SnellEnvelopePricer`,
/// `FiniteDifferencePricer` at [`OptionStyle::American`](crate::OptionStyle))
/// implement it too, and each says so on its own type. What the trait fixes
/// is the *query* shape — one spot, one strike, one rate, one dividend
/// yield, one maturity — not the exercise right.
///
/// # How pricing fails
///
/// This is the crate-wide convention for every pricer, Greek accessor and
/// surface builder, stated here once. Individual methods say *when* they hit
/// one of these cases; they do not restate the rule.
///
/// Pricing deliberately does **not** return [`Result`]. Threading `?` through
/// a strike/maturity grid would cost more than it buys, so the three failure
/// modes are separated by kind instead:
///
/// 1. **Invalid parameter — panic**, with a message naming the parameter and
///    its value (`"strike k must be strictly positive (got -1)"`). A
///    non-positive strike or a negative variance is programmer error, not a
///    market state, and there is no answer to return. `hagan_implied_vol` and
///    `SnellEnvelopePricer::validate_query` set the message style. A method
///    that can panic carries a `# Panics` section.
/// 2. **Not computable at this point — [`f64::NAN`], documented.** The inputs
///    are legitimate but the quantity is genuinely undefined *here*: a strike
///    outside a Fourier pricer's truncation grid, a second derivative at a
///    grid boundary, a central difference in $\tau$ that would step past
///    expiry, a yield at $\tau = 0$, a Greek this pricer does not expose. Any
///    method that can return `NaN` says so in its own doc.
/// 3. **Calibration did not converge — [`Result::Err`].** Calibration is the
///    one part of the crate with a fallible return channel, and it keeps it.
///
/// What the convention rules out is the fourth option: a **plausible-looking
/// sentinel**. Returning `0.0` from a failed volatility inversion hands the
/// caller a number that flows into an intrinsic-value price with nothing to
/// distinguish it from a real one, and `f64::max` will quietly discard a `NaN`
/// operand in favour of a finite one, so a clamp has to test for `NaN`
/// explicitly before it applies. `NaN` propagates; a zero does not.
///
/// Callers that need to *avoid* the `NaN` rather than detect it after the
/// fact should test first — `CarrMadanPricer::strike_in_grid` is the pattern.
pub trait ModelPricer {
  /// Price a call option.
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64;

  /// Price a put via European put-call parity.
  ///
  /// **Override this** unless the model's carry factor is literally
  /// $e^{-q\tau}$ and its exercise is European — the default silently
  /// returns a plausible wrong number otherwise. See `BSMPricer`
  /// (cost-of-carry conventions) and `BjerksundStensland2002Pricer`
  /// (American early exercise) for the two failure modes.
  ///
  /// This default is the one place the trait itself can produce the
  /// plausible-looking sentinel [the failure
  /// convention](ModelPricer#how-pricing-fails) otherwise rules out, and it
  /// cannot detect the mismatch: parity is
  /// arithmetic on a call price, so a wrong carry yields a finite,
  /// well-scaled, wrong put with no `NaN` to propagate and nothing to assert
  /// on. It stays a default because most implementors *are* European with
  /// carry $e^{-q\tau}$ — but a new implementor must decide, not inherit.
  /// `SuperSharePricer` shows the third option: override to return a
  /// documented `NaN` when the payoff has no put analogue at all.
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
/// Those defaults are case 2 of [the failure
/// convention](ModelPricer#how-pricing-fails), and the reason they are `NAN`
/// rather than `0.0` is that a Greek genuinely *is* zero sometimes — the vega
/// of a deep in-the-money digital, the gamma of a forward. A zero default
/// would make "this pricer does not expose vega" indistinguishable from
/// "vega is zero here", and the two call for opposite responses.
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
