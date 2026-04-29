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
//! Reference: F. Black, "The Pricing of Commodity Contracts", Journal of
//! Financial Economics 3(1-2), 167-179 (1976).
//!
//! Reference: D. Brigo & F. Mercurio, "Interest Rate Models — Theory and
//! Practice", Springer, 2nd ed. (2006), §1.5.

use stochastic_rs_distributions::special::norm_cdf;

use super::types::InterestRateOptionKind;
use super::types::VolatilityQuoteKind;
use super::volatility::VolatilityModel;
use crate::traits::FloatExt;

/// Black-76 caplet undiscounted forward value $F\Phi(d_1)-K\Phi(d_2)$.
///
/// Returns zero if any input makes the formula undefined (non-positive
/// forward/strike/time or non-positive vol).
pub fn black_forward_caplet<T: FloatExt>(forward: T, strike: T, tau: T, sigma: T) -> T {
  let f = forward.to_f64().unwrap_or(0.0);
  let k = strike.to_f64().unwrap_or(0.0);
  let t = tau.to_f64().unwrap_or(0.0);
  let v = sigma.to_f64().unwrap_or(0.0);
  if f <= 0.0 || k <= 0.0 || t <= 0.0 || v <= 0.0 {
    return T::from_f64_fast((f - k).max(0.0));
  }
  let sqrt_t = t.sqrt();
  let d1 = ((f / k).ln() + 0.5 * v * v * t) / (v * sqrt_t);
  let d2 = d1 - v * sqrt_t;
  T::from_f64_fast(f * norm_cdf(d1) - k * norm_cdf(d2))
}

/// Black-76 floorlet undiscounted forward value $K\Phi(-d_2)-F\Phi(-d_1)$.
pub fn black_forward_floorlet<T: FloatExt>(forward: T, strike: T, tau: T, sigma: T) -> T {
  let f = forward.to_f64().unwrap_or(0.0);
  let k = strike.to_f64().unwrap_or(0.0);
  let t = tau.to_f64().unwrap_or(0.0);
  let v = sigma.to_f64().unwrap_or(0.0);
  if f <= 0.0 || k <= 0.0 || t <= 0.0 || v <= 0.0 {
    return T::from_f64_fast((k - f).max(0.0));
  }
  let sqrt_t = t.sqrt();
  let d1 = ((f / k).ln() + 0.5 * v * v * t) / (v * sqrt_t);
  let d2 = d1 - v * sqrt_t;
  T::from_f64_fast(k * norm_cdf(-d2) - f * norm_cdf(-d1))
}

/// Bachelier caplet undiscounted forward value.
pub fn bachelier_forward_caplet<T: FloatExt>(forward: T, strike: T, tau: T, sigma: T) -> T {
  let f = forward.to_f64().unwrap_or(0.0);
  let k = strike.to_f64().unwrap_or(0.0);
  let t = tau.to_f64().unwrap_or(0.0);
  let v = sigma.to_f64().unwrap_or(0.0);
  if t <= 0.0 || v <= 0.0 {
    return T::from_f64_fast((f - k).max(0.0));
  }
  let sqrt_vt = v * t.sqrt();
  let d = (f - k) / sqrt_vt;
  let pdf = (-0.5 * d * d).exp() / (2.0 * std::f64::consts::PI).sqrt();
  T::from_f64_fast((f - k) * norm_cdf(d) + sqrt_vt * pdf)
}

/// Bachelier floorlet undiscounted forward value.
pub fn bachelier_forward_floorlet<T: FloatExt>(forward: T, strike: T, tau: T, sigma: T) -> T {
  let f = forward.to_f64().unwrap_or(0.0);
  let k = strike.to_f64().unwrap_or(0.0);
  let t = tau.to_f64().unwrap_or(0.0);
  let v = sigma.to_f64().unwrap_or(0.0);
  if t <= 0.0 || v <= 0.0 {
    return T::from_f64_fast((k - f).max(0.0));
  }
  let sqrt_vt = v * t.sqrt();
  let d = (f - k) / sqrt_vt;
  let pdf = (-0.5 * d * d).exp() / (2.0 * std::f64::consts::PI).sqrt();
  T::from_f64_fast((k - f) * norm_cdf(-d) + sqrt_vt * pdf)
}

/// Caplet or floorlet price discounted to the valuation date.
///
/// `tau` is the year fraction from valuation date to the caplet expiry (the
/// fixing date of the forward rate). `accrual_factor` is $\alpha$, the day
/// count fraction for the coupon period. `discount_factor` is $P(t,t_p)$ to
/// the payment date $t_p$.
#[allow(clippy::too_many_arguments)]
pub fn caplet_price<T: FloatExt, V: VolatilityModel<T> + ?Sized>(
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
}
