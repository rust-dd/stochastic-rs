//! Bachelier's normal model, `dF = σ_N dW` with `σ_N` in price units, which prices negative strikes
//! and forwards; [`normal_implied_volatility`] inverts a forward price on the same formula.
//!
//! $$
//! c = e^{-r\tau}\left[(F - K)\,\Phi(d) + \sigma_N\sqrt{\tau}\,\varphi(d)\right],\qquad d = \frac{F - K}{\sigma_N\sqrt{\tau}},\qquad F = S e^{(r-q)\tau}
//! $$
//!
//! Reference: Bachelier (1900), Ann. Sci. Éc. Norm. Supér. 3(17), 21–86
//! Reference: Schachermayer & Teichmann (2008), Math. Finance 18(1), 155–170

use stochastic_rs_distributions::special::norm_pdf;

use crate::OptionType;
use crate::instruments::option::caplet::bachelier_forward_caplet;
use crate::instruments::option::caplet::bachelier_forward_floorlet;
use crate::pricing::digital::expired;
use crate::pricing::digital::vanilla_payoff;
use crate::pricing::exotic::assert_sigma;
use crate::traits::ModelPricer;
use crate::traits::VanillaEuropeanCall;

/// Bachelier (normal) pricer holding the normal volatility `σ_N`, quoted in
/// the underlying's price units per square-root year.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BachelierPricer {
  /// Normal volatility σ_N.
  pub sigma: f64,
}

impl BachelierPricer {
  /// Panics unless `sigma` is finite and `>= 0`; at `0` the forward is certain and the price is
  /// its discounted intrinsic value.
  pub fn new(sigma: f64) -> Self {
    assert_sigma(sigma);
    Self { sigma }
  }

  /// Forward `F = S e^{(r − q)τ}` the option is struck on.
  pub fn forward(&self, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    s * ((r - q) * tau).exp()
  }

  /// Normal vega `e^{−rτ} √τ φ(d)`, shared by calls and puts; NaN at and past expiry.
  pub fn vega(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if tau.is_nan() || tau <= 0.0 {
      return f64::NAN;
    }
    let forward = self.forward(s, r, q, tau);
    let sqrt_tau = tau.sqrt();
    let d = (forward - k) / (self.sigma * sqrt_tau);
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
    if let Some(value) = expired(tau, &[r, q], || vanilla_payoff(OptionType::Call, s, k)) {
      return value;
    }
    (-r * tau).exp() * bachelier_forward_caplet(self.forward(s, r, q, tau), k, tau, self.sigma)
  }

  /// The closed form $(K-F)\Phi(-d)+\sigma\sqrt\tau\,\varphi(d)$, not the parity default, which
  /// recomposes the put from the call.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if let Some(value) = expired(tau, &[r, q], || vanilla_payoff(OptionType::Put, s, k)) {
      return value;
    }
    (-r * tau).exp() * bachelier_forward_floorlet(self.forward(s, r, q, tau), k, tau, self.sigma)
  }
}

/// A European vanilla call on the default forward `Se^{(r − q)τ}`, so a Black inversion turns its
/// normal volatility into the lognormal one strike by strike.
impl VanillaEuropeanCall for BachelierPricer {}

/// Normal volatility of an undiscounted forward price by bracketed Newton on the monotone price, to
/// `1e-14` relative; NaN below intrinsic value, at `tau ≤ 0` or for a non-finite price.
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
  // `σ√τ/√(2π)`, the time value at the money, seeds the iteration and bounds `σ` below elsewhere.
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

  /// scipy 1.x at `F = 100e^{0.0225}`, `d = (F − 105)/(20√0.75)`: call `5.425620481357` and put
  /// `8.049840381737` by the closed form, to the references' twelve decimals.
  #[test]
  fn matches_the_scipy_reference() {
    let (s, k, r, q, tau) = Q;
    let (call, put) = (
      BachelierPricer::new(20.0).price_call(s, k, r, q, tau),
      BachelierPricer::new(20.0).price_put(s, k, r, q, tau),
    );
    assert!((call - 5.425620481357).abs() < 1e-10, "call {call}");
    assert!((put - 8.049840381737).abs() < 1e-10, "put {put}");
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
    let (call, put) = (
      model.price_call(s, k, r, q, tau),
      model.price_put(s, k, r, q, tau),
    );
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

  /// Both types round-trip at strikes `mσ√τ` from the forward, `|d| ≤ 4` (negative at `σ = 60`);
  /// past that the time value is below machine precision and identifies no volatility.
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

  /// At `σ_N = 0` the forward is certain, so the call is its discounted intrinsic value.
  #[test]
  fn a_zero_normal_volatility_prices_the_discounted_intrinsic() {
    let (s, _, r, q, tau) = Q;
    let model = BachelierPricer::new(0.0);
    let want = (-r * tau).exp() * (model.forward(s, r, q, tau) - 95.0);
    assert_eq!(model.price_call(s, 95.0, r, q, tau), want);
  }

  #[test]
  #[should_panic(
    expected = "sigma must satisfy `sigma.is_finite() && sigma >= 0.0`, got sigma = -20.0"
  )]
  fn new_rejects_a_negative_normal_volatility() {
    let _ = BachelierPricer::new(-20.0);
  }

  #[test]
  #[should_panic(
    expected = "sigma must satisfy `sigma.is_finite() && sigma >= 0.0`, got sigma = inf"
  )]
  fn new_rejects_an_infinite_normal_volatility() {
    let _ = BachelierPricer::new(f64::INFINITY);
  }

  #[test]
  #[should_panic(
    expected = "sigma must satisfy `sigma.is_finite() && sigma >= 0.0`, got sigma = NaN"
  )]
  fn new_rejects_a_nan_normal_volatility() {
    let _ = BachelierPricer::new(f64::NAN);
  }
}
