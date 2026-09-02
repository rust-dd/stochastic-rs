//! # Bachelier
//!
//! $$
//! c = e^{-r\tau}\left[(F - K)\,\Phi(d) + \sigma_N\sqrt{\tau}\,\varphi(d)\right],\qquad d = \frac{F - K}{\sigma_N\sqrt{\tau}},\qquad F = S e^{(r-q)\tau}
//! $$
//!
//! Normal (arithmetic Brownian) model for a spot underlying: the forward
//! diffuses as `dF = σ_N dW` with a volatility quoted in price units, so the
//! price is well defined at negative strikes and forwards and the smile is
//! flat in normal rather than lognormal volatility. `σ_N` is the model
//! state; the query `(s, k, r, q, τ)` travels as arguments like every
//! [`ModelPricer`]. [`normal_implied_volatility`] inverts an undiscounted
//! forward price on the same formula with a safeguarded Newton iteration,
//! which the pricer's [`BachelierPricer::implied_volatility`] wraps for a
//! spot quote; the forward-space functions are the ones the interest-rate
//! caplet/floorlet code already prices with.
//!
//! References: Bachelier, L. (1900), *Théorie de la spéculation*, Annales
//! scientifiques de l'É.N.S. 3(17), 21–86; Schachermayer, W. & Teichmann, J.
//! (2008), *How close are the option pricing formulas of Bachelier and
//! Black–Merton–Scholes?*, Mathematical Finance 18(1), 155–170.

use stochastic_rs_distributions::special::norm_pdf;

use crate::OptionType;
use crate::instruments::option::caplet::bachelier_forward_caplet;
use crate::instruments::option::caplet::bachelier_forward_floorlet;
use crate::traits::ModelPricer;
use crate::traits::VanillaEuropeanCall;

/// Bachelier (normal) pricer holding the normal volatility `σ_N`, quoted in
/// the underlying's price units per square-root year.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BachelierPricer {
  /// Normal volatility σ_N.
  pub v: f64,
}

impl BachelierPricer {
  pub const fn new(v: f64) -> Self {
    Self { v }
  }

  /// Forward `F = S e^{(r − q)τ}` the option is struck on.
  pub fn forward(&self, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    s * ((r - q) * tau).exp()
  }

  /// Call and put at the query `(s, k, r, q, tau)`.
  pub fn call_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> (f64, f64) {
    let forward = self.forward(s, r, q, tau);
    let disc = (-r * tau).exp();
    (
      disc * bachelier_forward_caplet(forward, k, tau, self.v),
      disc * bachelier_forward_floorlet(forward, k, tau, self.v),
    )
  }

  /// Normal vega `e^{−rτ} √τ φ(d)`, shared by calls and puts.
  pub fn vega(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let forward = self.forward(s, r, q, tau);
    let sqrt_tau = tau.sqrt();
    let d = (forward - k) / (self.v * sqrt_tau);
    (-r * tau).exp() * sqrt_tau * norm_pdf(d)
  }

  /// Normal volatility implied by a spot-quoted option `price`; `NaN` when
  /// the price sits below intrinsic value or the inputs are degenerate.
  pub fn implied_volatility(
    &self,
    price: f64,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    option_type: OptionType,
  ) -> f64 {
    let forward = self.forward(s, r, q, tau);
    let undiscounted = price * (r * tau).exp();
    normal_implied_volatility(undiscounted, forward, k, tau, option_type)
  }
}

impl ModelPricer for BachelierPricer {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.call_put(s, k, r, q, tau).0
  }

  /// The carry is `r − q`, so the trait's parity default would be right;
  /// the closed form is used anyway so the put is exact rather than
  /// recomposed from the call.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.call_put(s, k, r, q, tau).1
  }
}

/// A European vanilla call on `S` with the default forward `S e^{(r − q)τ}`,
/// so a Black inversion of its prices is meaningful: it converts the normal
/// volatility into the lognormal one strike by strike.
impl VanillaEuropeanCall for BachelierPricer {}

/// Normal volatility implied by an **undiscounted forward** option price on
/// forward `forward` and strike `strike`: safeguarded Newton on the
/// monotone Bachelier price, bisection-bracketed, to a relative tolerance
/// of `1e-14`. Returns `NaN` below intrinsic value, at `tau ≤ 0`, or for a
/// non-finite price.
pub fn normal_implied_volatility(
  price: f64,
  forward: f64,
  strike: f64,
  tau: f64,
  option_type: OptionType,
) -> f64 {
  if !(price.is_finite() && forward.is_finite() && strike.is_finite()) || tau <= 0.0 {
    return f64::NAN;
  }
  let intrinsic = match option_type {
    OptionType::Call => (forward - strike).max(0.0),
    OptionType::Put => (strike - forward).max(0.0),
  };
  if price < intrinsic {
    return f64::NAN;
  }
  let time_value = price - intrinsic;
  if time_value == 0.0 {
    return 0.0;
  }
  let sqrt_tau = tau.sqrt();
  let value = |sigma: f64| match option_type {
    OptionType::Call => bachelier_forward_caplet(forward, strike, tau, sigma),
    OptionType::Put => bachelier_forward_floorlet(forward, strike, tau, sigma),
  };
  // At the money the time value is σ√τ/√(2π); this seeds the iteration and
  // is a lower bound on σ everywhere else, since the time value falls as the
  // strike moves away from the forward at fixed σ.
  let mut lo = time_value * (2.0 * std::f64::consts::PI).sqrt() / sqrt_tau;
  let mut hi = lo;
  while value(hi) < price {
    hi *= 2.0;
    if !hi.is_finite() {
      return f64::NAN;
    }
  }
  let mut sigma = 0.5 * (lo + hi);
  for _ in 0..100 {
    let f = value(sigma) - price;
    if f.abs() <= 1e-14 * price.max(f64::MIN_POSITIVE) {
      return sigma;
    }
    if f > 0.0 {
      hi = sigma;
    } else {
      lo = sigma;
    }
    let d = (forward - strike) / (sigma * sqrt_tau);
    let vega = sqrt_tau * norm_pdf(d);
    let newton = sigma - f / vega;
    sigma = if vega > 0.0 && newton > lo && newton < hi {
      newton
    } else {
      0.5 * (lo + hi)
    };
    if (hi - lo).abs() <= 1e-15 * hi {
      return sigma;
    }
  }
  sigma
}

#[cfg(test)]
mod tests {
  use super::*;

  const Q: (f64, f64, f64, f64, f64) = (100.0, 105.0, 0.05, 0.02, 0.75);

  /// Reference: scipy 1.x, `F = 100 e^{0.0225}`, `d = (F − 105) / (20 √0.75)`,
  /// `call = e^{−0.0375} ((F − 105) Φ(d) + 20 √0.75 φ(d))` = 5.425620481357,
  /// `put = e^{−0.0375} ((105 − F) Φ(−d) + 20 √0.75 φ(d))` = 8.049840381737.
  /// The tolerance follows the crate's `norm_cdf` (Abramowitz–Stegun 7.1.26
  /// erf, 1.5e-7 absolute), which moves a price of this size by up to ~1e-5.
  #[test]
  fn matches_the_scipy_reference() {
    let (s, k, r, q, tau) = Q;
    let (call, put) = BachelierPricer::new(20.0).call_put(s, k, r, q, tau);
    assert!((call - 5.425620481357).abs() < 5e-5, "call {call}");
    assert!((put - 8.049840381737).abs() < 5e-5, "put {put}");
  }

  /// At the money the call is `e^{−rτ} σ_N √τ / √(2π)` exactly.
  #[test]
  fn at_the_money_price_is_the_normal_time_value() {
    let (s, _, r, q, tau) = Q;
    let model = BachelierPricer::new(20.0);
    let forward = model.forward(s, r, q, tau);
    let call = model.price_call(s, forward, r, q, tau);
    let want = (-r * tau).exp() * 20.0 * tau.sqrt() / (2.0 * std::f64::consts::PI).sqrt();
    assert!((call - want).abs() < 1e-9, "{call} vs {want}");
  }

  #[test]
  fn parity_and_trait_routing_hold() {
    let (s, k, r, q, tau) = Q;
    let model = BachelierPricer::new(20.0);
    let (call, put) = model.call_put(s, k, r, q, tau);
    let want = (-r * tau).exp() * (model.forward(s, r, q, tau) - k);
    assert!((call - put - want).abs() < 1e-9);
    assert_eq!(model.price_call(s, k, r, q, tau), call);
    assert_eq!(model.price_put(s, k, r, q, tau), put);
    assert_eq!(model.price_option(s, k, r, q, tau, OptionType::Put), put);
    assert_eq!(
      model.vanilla_call_forward(s, r, q, tau),
      model.forward(s, r, q, tau)
    );
    assert!(model.vega(s, k, r, q, tau) > 0.0);
  }

  /// Round trip across strikes placed at `m σ√τ` from the forward (`|d| ≤ 4`,
  /// so the time value stays resolvable in double precision; at `σ = 60`
  /// the far strikes are negative), for both option types. Beyond that
  /// band the time value falls below machine precision and no volatility
  /// is identifiable from the price.
  #[test]
  fn implied_normal_volatility_round_trips() {
    let (s, _, r, q, tau) = Q;
    for sigma in [5.0_f64, 20.0, 60.0] {
      let model = BachelierPricer::new(sigma);
      let forward = model.forward(s, r, q, tau);
      for m in [-4.0_f64, -1.0, 0.0, 0.5, 2.0, 4.0] {
        let k = forward + m * sigma * tau.sqrt();
        for option_type in [OptionType::Call, OptionType::Put] {
          let price = model.price_option(s, k, r, q, tau, option_type);
          let implied = model.implied_volatility(price, s, k, r, q, tau, option_type);
          assert!(
            (implied - sigma).abs() < 1e-9 * sigma,
            "sigma {sigma} k {k}: {implied}"
          );
        }
      }
    }
    assert!(BachelierPricer::new(60.0).forward(s, r, q, tau) - 4.0 * 60.0 * tau.sqrt() < 0.0);
  }

  #[test]
  fn implied_volatility_flags_prices_below_intrinsic() {
    let (s, k, r, q, tau) = Q;
    let model = BachelierPricer::new(20.0);
    let intrinsic = (-r * tau).exp() * (model.forward(s, r, q, tau) - k);
    assert!(
      model
        .implied_volatility(intrinsic - 1.0, s, k, r, q, tau, OptionType::Call)
        .is_nan()
    );
    assert!(normal_implied_volatility(1.0, 100.0, 100.0, 0.0, OptionType::Call).is_nan());
    assert_eq!(
      normal_implied_volatility(5.0, 105.0, 100.0, 1.0, OptionType::Call),
      0.0
    );
  }
}
