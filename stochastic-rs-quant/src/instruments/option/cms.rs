//! CMS caplet and floorlet pricing with Hagan's linear TSR convexity
//! adjustment.
//!
//! $$
//! S_{\mathrm{adj}}=S_0+S_0^{2}\,\sigma_{B}^{2}\,T_{\mathrm{fix}}\,\lambda(S_0) +
//!   S_0\,\sigma_{B}^{2}\,T_{\mathrm{fix}}\,\Delta\,\tfrac{S_0}{1+S_0\Delta},
//!   \qquad
//! \lambda(S_0)=\frac{S_0\,G'(S_0)}{G(S_0)},\;
//! G(S)=\frac{1-(1+S\delta)^{-N}}{S}
//! $$
//!
//! The first term is the natural-timing convexity (annuity-measure change),
//! the second the payment-delay correction. `lambda` is computed by central
//! finite differences on $G$ under a flat-yield assumption.
//!
//! Reference: Hagan, "Convexity Conundrums: Pricing CMS Swaps, Caps and
//! Floors", Wilmott Magazine (2003).
//!
//! Reference: Brigo & Mercurio, "Interest Rate Models — Theory and Practice",
//! Springer 2nd ed. (2006), §13.8.

use super::caplet::black_forward_caplet;
use super::caplet::black_forward_floorlet;
use super::volatility::VolatilityModel;
use crate::traits::RealExt;

/// Flat-yield annuity-to-par function $G(S)=(1-(1+S\delta)^{-N})/S$.
fn g_function<T: RealExt>(s: T, n: T, delta: T) -> T {
  if s.abs() <= T::from_f64_fast(1e-12) {
    return n * delta;
  }
  let discount = T::one() / (T::one() + s * delta);
  (T::one() - discount.powf(n)) / s
}

/// Hagan linear-TSR convexity factor $\lambda = S_0\,G'(S_0)/G(S_0)$.
///
/// `swap_years` is the tenor of the CMS-referenced swap; `fixed_freq` is the
/// number of fixed-leg payments per year of that swap.
pub fn hagan_linear_tsr_convexity_factor<T: RealExt>(s0: T, swap_years: T, fixed_freq: T) -> T {
  let delta = T::one() / fixed_freq;
  let n = swap_years * fixed_freq;
  let bump = s0.abs() * T::from_f64_fast(1e-4);
  let eps = if bump > T::from_f64_fast(1e-8) {
    bump
  } else {
    T::from_f64_fast(1e-8)
  };
  let g0 = g_function(s0, n, delta);
  if g0.abs() < T::from_f64_fast(1e-14) {
    return T::zero();
  }
  let g_plus = g_function(s0 + eps, n, delta);
  let g_minus = g_function(s0 - eps, n, delta);
  let g_prime = (g_plus - g_minus) / (T::from_f64_fast(2.0) * eps);
  s0 * g_prime / g0
}

/// Hagan convexity-adjusted forward CMS rate.
///
/// `payment_delay` is $T_{pay}-T_{fix}$ in years, zero or positive; the delay
/// term is only applied when strictly positive.
pub fn hagan_linear_tsr_convexity_adjustment<T: RealExt>(
  s0: T,
  sigma_black: T,
  t_fix: T,
  swap_years: T,
  fixed_freq: T,
  payment_delay: T,
) -> T {
  let lambda = hagan_linear_tsr_convexity_factor(s0, swap_years, fixed_freq);
  let variance = sigma_black * sigma_black * t_fix;
  let natural = s0 * s0 * variance * lambda;
  let delay = if payment_delay > T::zero() {
    s0 * variance * payment_delay * s0 / (T::one() + s0 * payment_delay)
  } else {
    T::zero()
  };
  natural + delay
}

/// CMS caplet struck at `strike` with Hagan convexity adjustment applied to
/// the forward CMS rate before Black-76 pricing.
#[derive(Debug, Clone)]
pub struct CmsCaplet<T: RealExt, V: VolatilityModel<T>> {
  /// Fixed strike $K$.
  pub strike: T,
  /// Coupon notional.
  pub notional: T,
  /// Coupon accrual factor $\alpha$.
  pub accrual_factor: T,
  /// Discount factor to the payment date $P(0,T_{pay})$.
  pub discount_factor: T,
  /// Forward CMS rate $S_0$.
  pub forward_cms: T,
  /// Year fraction from valuation to fixing $T_{fix}$.
  pub t_fix: T,
  /// Swap tenor in years.
  pub swap_years: T,
  /// Fixed-leg frequency of the referenced swap (payments per year).
  pub fixed_freq: T,
  /// Payment delay $T_{pay}-T_{fix}$ in years (0 for natural timing).
  pub payment_delay: T,
  /// Volatility model for $\sigma_B(S_0,K,T_{fix})$.
  pub vol: V,
}

impl<T: RealExt, V: VolatilityModel<T>> CmsCaplet<T, V> {
  /// Present value.
  pub fn price(&self) -> T {
    price_cms_payoff(
      self.strike,
      self.notional,
      self.accrual_factor,
      self.discount_factor,
      self.forward_cms,
      self.t_fix,
      self.swap_years,
      self.fixed_freq,
      self.payment_delay,
      &self.vol,
      true,
    )
  }
}

/// CMS floorlet — put on the convexity-adjusted forward CMS rate.
#[derive(Debug, Clone)]
pub struct CmsFloorlet<T: RealExt, V: VolatilityModel<T>> {
  /// Fixed strike $K$.
  pub strike: T,
  /// Coupon notional.
  pub notional: T,
  /// Coupon accrual factor.
  pub accrual_factor: T,
  /// Discount factor to payment date.
  pub discount_factor: T,
  /// Forward CMS rate.
  pub forward_cms: T,
  /// Year fraction to fixing.
  pub t_fix: T,
  /// Swap tenor in years.
  pub swap_years: T,
  /// Fixed-leg frequency.
  pub fixed_freq: T,
  /// Payment delay in years.
  pub payment_delay: T,
  /// Volatility model.
  pub vol: V,
}

impl<T: RealExt, V: VolatilityModel<T>> CmsFloorlet<T, V> {
  /// Present value.
  pub fn price(&self) -> T {
    price_cms_payoff(
      self.strike,
      self.notional,
      self.accrual_factor,
      self.discount_factor,
      self.forward_cms,
      self.t_fix,
      self.swap_years,
      self.fixed_freq,
      self.payment_delay,
      &self.vol,
      false,
    )
  }
}

/// A fixing at `t_fix <= 0` is known, so it pays its intrinsic value on `s0` without a vol or a convexity adjustment.
#[allow(clippy::too_many_arguments)]
fn price_cms_payoff<T: RealExt, V: VolatilityModel<T> + ?Sized>(
  strike: T,
  notional: T,
  accrual: T,
  discount: T,
  s0: T,
  t_fix: T,
  swap_years: T,
  fixed_freq: T,
  payment_delay: T,
  vol: &V,
  is_caplet: bool,
) -> T {
  if t_fix <= T::zero() {
    let moneyness = if is_caplet { s0 - strike } else { strike - s0 };
    return notional * accrual * discount * moneyness.max_or_nan(T::zero());
  }
  let sigma = vol.implied_volatility(s0, strike, t_fix);
  let ca =
    hagan_linear_tsr_convexity_adjustment(s0, sigma, t_fix, swap_years, fixed_freq, payment_delay);
  let s_adj = s0 + ca;
  let forward_value = if is_caplet {
    black_forward_caplet(s_adj, strike, t_fix, sigma)
  } else {
    black_forward_floorlet(s_adj, strike, t_fix, sigma)
  };
  notional * accrual * discount * forward_value
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::instruments::option::volatility::SabrVolatility;

  #[test]
  fn a_known_fixing_pays_its_intrinsic_value_without_the_vol_model() {
    let vol = SabrVolatility::new(0.035_f64, 0.5, 0.4, -0.2);
    let caplet = |t_fix: f64| {
      CmsCaplet {
        strike: 0.02,
        notional: 1e6,
        accrual_factor: 1.0,
        discount_factor: 0.97,
        forward_cms: 0.03,
        t_fix,
        swap_years: 10.0,
        fixed_freq: 2.0,
        payment_delay: 0.0,
        vol,
      }
      .price()
    };
    let floorlet = |t_fix: f64| {
      CmsFloorlet {
        strike: 0.04,
        notional: 1e6,
        accrual_factor: 1.0,
        discount_factor: 0.97,
        forward_cms: 0.03,
        t_fix,
        swap_years: 10.0,
        fixed_freq: 2.0,
        payment_delay: 0.0,
        vol,
      }
      .price()
    };
    for t_fix in [-0.5, 0.0] {
      assert!((caplet(t_fix) - 9700.0).abs() < 1e-6, "t_fix = {t_fix}");
      assert!((floorlet(t_fix) - 9700.0).abs() < 1e-6, "t_fix = {t_fix}");
    }
    assert!(caplet(0.5).is_finite() && caplet(0.5) != caplet(0.0));
  }

  #[test]
  fn convexity_factor_is_negative_for_standard_swap() {
    let lambda = hagan_linear_tsr_convexity_factor(0.04_f64, 10.0, 2.0);
    assert!(lambda < 0.0, "lambda should be negative, got {lambda}");
    assert!(lambda.is_finite());
  }

  #[test]
  fn convexity_factor_goes_to_zero_for_short_swap() {
    let lambda_long = hagan_linear_tsr_convexity_factor(0.04_f64, 20.0, 2.0).abs();
    let lambda_short = hagan_linear_tsr_convexity_factor(0.04_f64, 1.0, 2.0).abs();
    assert!(
      lambda_long > lambda_short,
      "longer swap must show stronger convexity, long={lambda_long} short={lambda_short}"
    );
  }

  #[test]
  fn payment_delay_adds_positive_adjustment() {
    let nat = hagan_linear_tsr_convexity_adjustment(0.04_f64, 0.3, 2.0, 10.0, 2.0, 0.0);
    let delayed = hagan_linear_tsr_convexity_adjustment(0.04_f64, 0.3, 2.0, 10.0, 2.0, 0.5);
    assert!(delayed > nat, "delay must push adjustment upwards");
  }
}
