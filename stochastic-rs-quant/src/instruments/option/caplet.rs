//! Closed-form caplet and floorlet prices under Black-76 and Bachelier.
//!
//! $$
//! \mathrm{Caplet} = N\,\alpha\,P(t,t_p)\,\bigl[F\,\Phi(d_1) - K\,\Phi(d_2)\bigr],\quad
//! d_{1,2}=\frac{\ln(F/K)\pm\tfrac12\sigma^2\tau}{\sigma\sqrt{\tau}}
//! $$
//! $$
//! \mathrm{Caplet}_{\mathrm{Bach}} = N\,\alpha\,P(t,t_p)\,
//!   \bigl[(F-K)\,\Phi(d) + \sigma\sqrt{\tau}\,\varphi(d)\bigr],\quad
//!   d=\frac{F-K}{\sigma\sqrt{\tau}}
//! $$
//!
//! Reference: Black (1976), "The Pricing of Commodity Contracts", J. Financial Economics 3(1-2)
//! Reference: Brigo & Mercurio (2006), "Interest Rate Models", 2nd ed., §1.5

use stochastic_rs_distributions::special::norm_cdf;

use super::types::InterestRateOptionKind;
use super::types::VolatilityQuoteKind;
use super::volatility::VolatilityModel;
use crate::traits::RealExt;

/// Payoff where the closed form is undefined; NaN for a NaN rate or time, or a NaN vol while
/// the option is still live.
fn intrinsic(moneyness: f64, t: f64, v: f64) -> f64 {
  if t.is_nan() || (t > 0.0 && v.is_nan()) {
    f64::NAN
  } else {
    moneyness.max_or_nan(0.0)
  }
}

/// Black-76 caplet forward value `FΦ(d₁) − KΦ(d₂)`, intrinsic at `F`, `K`, `τ` or `σ` ≤ 0; NaN for
/// a NaN rate or time and, while live, a NaN `σ` or `F, K < 0`: shift them or use Bachelier there.
pub fn black_forward_caplet<T: RealExt>(forward: T, strike: T, tau: T, sigma: T) -> T {
  let f = forward.to_f64().unwrap_or(0.0);
  let k = strike.to_f64().unwrap_or(0.0);
  let t = tau.to_f64().unwrap_or(0.0);
  let v = sigma.to_f64().unwrap_or(0.0);
  if f <= 0.0 || k <= 0.0 || t <= 0.0 || v <= 0.0 {
    if f < 0.0 && k < 0.0 && t > 0.0 && v > 0.0 {
      return T::nan();
    }
    return T::from_f64_fast(intrinsic(f - k, t, v));
  }
  let sqrt_t = t.sqrt();
  let d1 = ((f / k).ln() + 0.5 * v * v * t) / (v * sqrt_t);
  let d2 = d1 - v * sqrt_t;
  T::from_f64_fast(f * norm_cdf(d1) - k * norm_cdf(d2))
}

/// Black-76 floorlet forward value `KΦ(−d₂) − FΦ(−d₁)`, with [`black_forward_caplet`]'s edge cases.
pub fn black_forward_floorlet<T: RealExt>(forward: T, strike: T, tau: T, sigma: T) -> T {
  let f = forward.to_f64().unwrap_or(0.0);
  let k = strike.to_f64().unwrap_or(0.0);
  let t = tau.to_f64().unwrap_or(0.0);
  let v = sigma.to_f64().unwrap_or(0.0);
  if f <= 0.0 || k <= 0.0 || t <= 0.0 || v <= 0.0 {
    if f < 0.0 && k < 0.0 && t > 0.0 && v > 0.0 {
      return T::nan();
    }
    return T::from_f64_fast(intrinsic(k - f, t, v));
  }
  let sqrt_t = t.sqrt();
  let d1 = ((f / k).ln() + 0.5 * v * v * t) / (v * sqrt_t);
  let d2 = d1 - v * sqrt_t;
  T::from_f64_fast(k * norm_cdf(-d2) - f * norm_cdf(-d1))
}

/// Bachelier caplet forward value, any sign of `F` and `K`; intrinsic at `τ` or `σ` ≤ 0; NaN for a
/// NaN rate or time and, while live, a NaN `σ`.
pub fn bachelier_forward_caplet<T: RealExt>(forward: T, strike: T, tau: T, sigma: T) -> T {
  let f = forward.to_f64().unwrap_or(0.0);
  let k = strike.to_f64().unwrap_or(0.0);
  let t = tau.to_f64().unwrap_or(0.0);
  let v = sigma.to_f64().unwrap_or(0.0);
  if t <= 0.0 || v <= 0.0 {
    return T::from_f64_fast(intrinsic(f - k, t, v));
  }
  let sqrt_vt = v * t.sqrt();
  let d = (f - k) / sqrt_vt;
  let pdf = (-0.5 * d * d).exp() / (2.0 * std::f64::consts::PI).sqrt();
  T::from_f64_fast((f - k) * norm_cdf(d) + sqrt_vt * pdf)
}

/// Bachelier floorlet undiscounted forward value, with [`bachelier_forward_caplet`]'s fallback.
pub fn bachelier_forward_floorlet<T: RealExt>(forward: T, strike: T, tau: T, sigma: T) -> T {
  let f = forward.to_f64().unwrap_or(0.0);
  let k = strike.to_f64().unwrap_or(0.0);
  let t = tau.to_f64().unwrap_or(0.0);
  let v = sigma.to_f64().unwrap_or(0.0);
  if t <= 0.0 || v <= 0.0 {
    return T::from_f64_fast(intrinsic(k - f, t, v));
  }
  let sqrt_vt = v * t.sqrt();
  let d = (f - k) / sqrt_vt;
  let pdf = (-0.5 * d * d).exp() / (2.0 * std::f64::consts::PI).sqrt();
  T::from_f64_fast((k - f) * norm_cdf(-d) + sqrt_vt * pdf)
}

/// Caplet or floorlet price at the valuation date: `tau` runs to the forward's fixing date,
/// `accrual_factor` is $\alpha$ and `discount_factor` is $P(t,t_p)$ to the payment date.
#[allow(clippy::too_many_arguments)]
pub fn caplet_price<T: RealExt, V: VolatilityModel<T> + ?Sized>(
  forward: T,
  strike: T,
  tau: T,
  notional: T,
  accrual_factor: T,
  discount_factor: T,
  vol: &V,
  kind: InterestRateOptionKind,
) -> T {
  let sigma = vol.implied_volatility(forward, strike, tau);
  let forward_value = match (vol.quote_kind(), kind) {
    (VolatilityQuoteKind::Lognormal, InterestRateOptionKind::Cap) => {
      black_forward_caplet(forward, strike, tau, sigma)
    }
    (VolatilityQuoteKind::Lognormal, InterestRateOptionKind::Floor) => {
      black_forward_floorlet(forward, strike, tau, sigma)
    }
    (VolatilityQuoteKind::Normal, InterestRateOptionKind::Cap) => {
      bachelier_forward_caplet(forward, strike, tau, sigma)
    }
    (VolatilityQuoteKind::Normal, InterestRateOptionKind::Floor) => {
      bachelier_forward_floorlet(forward, strike, tau, sigma)
    }
  };
  notional * accrual_factor * discount_factor * forward_value
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::instruments::option::volatility::SabrVolatility;

  #[test]
  fn a_nan_rate_or_a_live_nan_vol_is_nan_not_a_clamped_value() {
    assert!(black_forward_caplet(f64::NAN, 0.02, -0.5, 0.2).is_nan());
    assert!(black_forward_floorlet(f64::NAN, 0.04, -0.5, 0.2).is_nan());
    assert!(bachelier_forward_caplet(f64::NAN, 0.02, 0.0, 0.01).is_nan());
    assert!(bachelier_forward_floorlet(0.03, f64::NAN, 0.0, 0.01).is_nan());
    assert!(black_forward_caplet(-0.01, 0.02, 1.0, f64::NAN).is_nan());
    assert_eq!(
      black_forward_caplet(0.03, 0.02, -0.5, f64::NAN),
      0.03 - 0.02
    );
  }

  #[test]
  fn a_sabr_vol_outside_its_domain_prices_a_live_caplet_as_nan() {
    let sabr = SabrVolatility::new(0.035_f64, 0.5, -0.2, 0.4);
    let cap = |forward: f64, tau: f64| {
      caplet_price(
        forward,
        0.02,
        tau,
        1e6,
        1.0,
        0.97,
        &sabr,
        InterestRateOptionKind::Cap,
      )
    };
    assert!(cap(-0.002, 1.0).is_nan());
    assert!((cap(0.03, -0.5) - 9700.0).abs() < 1e-6);
  }

  #[test]
  fn black_put_call_parity() {
    let f = 0.04_f64;
    let k = 0.035_f64;
    let tau = 1.5_f64;
    let sigma = 0.25_f64;
    let call = black_forward_caplet(f, k, tau, sigma);
    let put = black_forward_floorlet(f, k, tau, sigma);
    let parity = call - put;
    assert!((parity - (f - k)).abs() < 1e-12);
  }

  #[test]
  fn bachelier_put_call_parity() {
    let f = 0.04_f64;
    let k = 0.03_f64;
    let tau = 2.0_f64;
    let sigma = 0.0125_f64;
    let call = bachelier_forward_caplet(f, k, tau, sigma);
    let put = bachelier_forward_floorlet(f, k, tau, sigma);
    assert!(((call - put) - (f - k)).abs() < 1e-12);
  }

  #[test]
  fn bachelier_atm_closed_form() {
    let f = 0.03_f64;
    let tau = 1.0_f64;
    let sigma = 0.01_f64;
    let call = bachelier_forward_caplet(f, f, tau, sigma);
    let expected = sigma * tau.sqrt() / (2.0 * std::f64::consts::PI).sqrt();
    assert!((call - expected).abs() < 1e-12);
  }

  /// Black (1976) against mpmath's 50-digit lognormal expectation, the exact limits, and the one
  /// undefined case: a live option on a negative forward and a negative strike.
  #[test]
  fn black_76_matches_the_reference_keeps_its_limits_and_is_nan_on_two_negative_rates() {
    for ((f, k, t, v), (caplet, floorlet)) in [
      (
        (0.04_f64, 0.035, 1.5, 0.25),
        (0.007_485_867_418_647_512, 0.002_485_867_418_647_512),
      ),
      (
        (0.03, 0.02, 1.0, 0.2),
        (0.010_038_495_064_659_41, 0.000_038_495_064_659_410_43),
      ),
      (
        (0.02, 0.045, 0.5, 0.4),
        (0.000_005_101_911_912_307_895, 0.025_005_101_911_912_308),
      ),
    ] {
      assert!(
        (black_forward_caplet(f, k, t, v) - caplet).abs() < 1e-15,
        "caplet {f} {k}"
      );
      assert!(
        (black_forward_floorlet(f, k, t, v) - floorlet).abs() < 1e-15,
        "floorlet {f} {k}"
      );
    }
    assert_eq!(
      black_forward_caplet(0.03, -0.01, 1.0, 0.2),
      0.04,
      "k <= 0 < f: the payoff is linear"
    );
    assert_eq!(black_forward_floorlet(0.03, -0.01, 1.0, 0.2), 0.0);
    assert_eq!(
      black_forward_caplet(0.0, 0.02, 1.0, 0.2),
      0.0,
      "f = 0 is absorbed"
    );
    assert_eq!(black_forward_floorlet(0.0, 0.02, 1.0, 0.2), 0.02);
    assert_eq!(
      black_forward_caplet(-0.01, 0.02, 1.0, 0.2),
      0.0,
      "f < 0 < k under the sign-preserving reading"
    );
    assert_eq!(black_forward_floorlet(-0.01, 0.02, 1.0, 0.2), 0.03);
    assert_eq!(
      black_forward_caplet(-0.01, -0.02, 0.0, 0.2),
      0.01,
      "expired: intrinsic whatever the signs"
    );
    assert_eq!(
      black_forward_caplet(-0.01, -0.02, 1.0, 0.0),
      0.01,
      "zero vol: intrinsic"
    );
    assert!(
      black_forward_caplet(-0.01_f64, -0.02, 1.0, 0.2).is_nan(),
      "a live option on two negative rates has no Black-76 value"
    );
    assert!(black_forward_floorlet(-0.01_f64, -0.02, 1.0, 0.2).is_nan());
  }
}
