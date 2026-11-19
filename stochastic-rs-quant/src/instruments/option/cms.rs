//! CMS caplets and floorlets by Hagan's standard model: the swaption value plus a first-order
//! convexity correction, with the bond-math `G` linearised at the forward swap rate.
//!
//! $$
//! G(S)=\frac{S}{(1+S/q)^{\Delta}}\,\frac{1}{1-(1+S/q)^{-n}},\qquad
//! \frac{V_{\mathrm{cap}}}{D(t_p)}=\frac{C(K)}{L_0}
//!   +\frac{G'(S_0)}{G(S_0)}\,\mathbb{E}\bigl[(S-S_0)(S-K)^+\bigr]
//! $$
//!
//! Reference: Hagan (2003), "Convexity Conundrums: Pricing CMS Swaps, Caps, and Floors", Wilmott

use stochastic_rs_distributions::special::norm_cdf;

use super::caplet::bachelier_forward_caplet;
use super::caplet::bachelier_forward_floorlet;
use super::caplet::black_forward_caplet;
use super::caplet::black_forward_floorlet;
use super::types::VolatilityQuoteKind;
use super::volatility::VolatilityModel;
use crate::calendar::Frequency;
use crate::traits::RealExt;

/// $G'(S)/G(S)$ of Hagan's (2.13a), `n` periods of `delta` years paid `lag` periods on; by its
/// series below $|nS\delta| = 10^{-3}$, where the closed form cancels; NaN unless $n, 1+S\delta>0$.
fn log_slope<T: RealExt>(s: T, n: T, delta: T, lag: T) -> T {
  let c = T::from_f64_fast;
  let one = T::one();
  let x = s * delta;
  if !(n > T::zero() && one + x > T::zero()) {
    return T::nan();
  }
  let per_x = if (n * x).abs() < c(1e-3) {
    let c0 = (n + one) / c(2.0) - lag;
    let c1 = lag - (n + one) * (n + c(5.0)) / c(12.0);
    let c2 = (n + one) * (n + c(3.0)) / c(8.0) - lag;
    let c3 = lag + (n + one) * (n * n * n - n * n - c(109.0) * n - c(251.0)) / c(720.0);
    c0 + x * (c1 + x * (c2 + x * c3))
  } else {
    (one - (x * lag + n * x / (n * x.ln_1p()).exp_m1()) / (one + x)) / x
  };
  delta * per_x
}

/// `(n, δ, Δ)` of (2.13a) and (2.13b) for a swap of `swap_years` paid `payment_delay` years on.
fn periods<T: RealExt>(swap_years: T, fixed_freq: Frequency, payment_delay: T) -> (T, T, T) {
  let per_year = T::from_f64_fast(f64::from(fixed_freq.periods_per_year()));
  (
    swap_years * per_year,
    T::one() / per_year,
    payment_delay * per_year,
  )
}

/// Hagan's convexity factor $S_0\,G'(S_0)/G(S_0)$ of the standard model (2.13a), for a payment
/// `payment_delay` years after the swap starts; NaN unless `swap_years` and $1+S_0/q$ are positive.
pub fn hagan_linear_tsr_convexity_factor<T: RealExt>(
  s0: T,
  swap_years: T,
  fixed_freq: Frequency,
  payment_delay: T,
) -> T {
  let (n, delta, lag) = periods(swap_years, fixed_freq, payment_delay);
  s0 * log_slope(s0, n, delta, lag)
}

/// The rate a CMS swaplet adds to the forward under Black's model, Hagan's (3.5b) over
/// $D(t_p)$; `payment_delay` is $T_{pay}-T_{fix}$ in years.
pub fn hagan_linear_tsr_convexity_adjustment<T: RealExt>(
  s0: T,
  sigma_black: T,
  t_fix: T,
  swap_years: T,
  fixed_freq: Frequency,
  payment_delay: T,
) -> T {
  let factor = hagan_linear_tsr_convexity_factor(s0, swap_years, fixed_freq, payment_delay);
  s0 * factor * (sigma_black * sigma_black * t_fix).exp_m1()
}

/// A CMS caplet's or floorlet's value with the rate it is priced on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CmsValuation<T: RealExt> {
  /// Forward swap rate before the adjustment.
  pub forward_swap_rate: T,
  /// What a CMS swaplet adds to the forward, (3.5b) or (3.6b) at the at-the-money vol; zero once
  /// the rate has fixed.
  pub convexity_adjustment: T,
  pub npv: T,
}

/// CMS caplet: Hagan's (3.5c), or (3.6c) under a normal vol, at the strike's vol; in the money,
/// the swaplet at the at-the-money vol plus the floorlet, by (2.21b).
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
  /// Fixed-leg frequency of the referenced swap.
  pub fixed_freq: Frequency,
  /// Payment delay $T_{pay}-T_{fix}$ in years (0 for natural timing).
  pub payment_delay: T,
  /// Volatility model for $\sigma(S_0,K,T_{fix})$, lognormal or normal by its quote kind.
  pub vol: V,
}

impl<T: RealExt, V: VolatilityModel<T>> CmsCaplet<T, V> {
  /// Present value; under a lognormal vol, a forward at or below 0 is read as
  /// [`black_forward_caplet`] reads it.
  pub fn npv(&self) -> T {
    self.valuation().npv
  }

  /// Present value with the forward swap rate and its convexity adjustment.
  pub fn valuation(&self) -> CmsValuation<T> {
    cms_valuation(
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

/// CMS floorlet: Hagan's (3.5d), or (3.6d) under a normal vol, with [`CmsCaplet`]'s parity rule.
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
  pub fixed_freq: Frequency,
  /// Payment delay in years.
  pub payment_delay: T,
  /// Volatility model.
  pub vol: V,
}

impl<T: RealExt, V: VolatilityModel<T>> CmsFloorlet<T, V> {
  /// Present value; under a lognormal vol, a forward at or below 0 is read as
  /// [`black_forward_floorlet`] reads it.
  pub fn npv(&self) -> T {
    self.valuation().npv
  }

  /// Present value with the forward swap rate and its convexity adjustment.
  pub fn valuation(&self) -> CmsValuation<T> {
    cms_valuation(
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

/// Hagan's first-order formulas over $D(t_p)$ at one forward, fixing time and $G'(S_0)/G(S_0)$.
struct FirstOrder {
  kind: VolatilityQuoteKind,
  s: f64,
  t: f64,
  slope: f64,
}

impl FirstOrder {
  /// `σ²τ`, zero at a vol that is not positive, as the Black and Bachelier values treat it.
  fn variance(&self, sigma: f64) -> f64 {
    if sigma <= 0.0 {
      0.0
    } else {
      sigma * sigma * self.t
    }
  }

  /// The swaplet's rate over the forward, (3.5b) or (3.6b).
  fn swaplet(&self, sigma: f64) -> f64 {
    let v = self.variance(sigma);
    match self.kind {
      VolatilityQuoteKind::Lognormal => self.slope * self.s * self.s * v.exp_m1(),
      VolatilityQuoteKind::Normal => self.slope * v,
    }
  }

  /// $E[(S-S_0)(S-K)^+]$ for `sign = 1` and $E[(S_0-S)(K-S)^+]$ for `sign = -1`: the brackets of
  /// (3.5c) and (3.5d), or $a^2\tau\,N(\pm(S_0-K)/a\sqrt\tau)$ of (3.6c) and (3.6d).
  fn quadratic_payoff(&self, k: f64, sigma: f64, sign: f64) -> f64 {
    let (s, v) = (self.s, self.variance(sigma));
    if v == 0.0 {
      return 0.0;
    }
    let root = v.sqrt();
    match self.kind {
      VolatilityQuoteKind::Lognormal => {
        let moneyness = if k > 0.0 { (s / k).ln() } else { f64::INFINITY };
        let n = |lambda: f64| norm_cdf(sign * (moneyness + lambda * v) / root);
        s * s * v.exp() * n(1.5) - s * (s + k) * n(0.5) + s * k * n(-0.5)
      }
      VolatilityQuoteKind::Normal => v * norm_cdf(sign * (s - k) / root),
    }
  }

  /// (3.5c) or (3.6c).
  fn caplet(&self, k: f64, sigma: f64) -> f64 {
    let swaption = match self.kind {
      VolatilityQuoteKind::Lognormal => black_forward_caplet(self.s, k, self.t, sigma),
      VolatilityQuoteKind::Normal => bachelier_forward_caplet(self.s, k, self.t, sigma),
    };
    swaption + self.slope * self.quadratic_payoff(k, sigma, 1.0)
  }

  /// (3.5d) or (3.6d).
  fn floorlet(&self, k: f64, sigma: f64) -> f64 {
    let swaption = match self.kind {
      VolatilityQuoteKind::Lognormal => black_forward_floorlet(self.s, k, self.t, sigma),
      VolatilityQuoteKind::Normal => bachelier_forward_floorlet(self.s, k, self.t, sigma),
    };
    swaption - self.slope * self.quadratic_payoff(k, sigma, -1.0)
  }
}

/// A fixing at `t_fix <= 0` is known, so it pays its intrinsic value on `s0` without a vol or
/// a convexity adjustment.
#[allow(clippy::too_many_arguments)]
fn cms_valuation<T: RealExt, V: VolatilityModel<T> + ?Sized>(
  strike: T,
  notional: T,
  accrual: T,
  discount: T,
  s0: T,
  t_fix: T,
  swap_years: T,
  fixed_freq: Frequency,
  payment_delay: T,
  vol: &V,
  is_caplet: bool,
) -> CmsValuation<T> {
  if t_fix <= T::zero() {
    let moneyness = if is_caplet { s0 - strike } else { strike - s0 };
    return CmsValuation {
      forward_swap_rate: s0,
      convexity_adjustment: T::zero(),
      npv: notional * accrual * discount * moneyness.max_or_nan(T::zero()),
    };
  }
  let real = |x: T| x.to_f64().unwrap_or(f64::NAN);
  let (n, delta, lag) = periods(swap_years, fixed_freq, payment_delay);
  let (s, k) = (real(s0), real(strike));
  let model = FirstOrder {
    kind: vol.quote_kind(),
    s,
    t: real(t_fix),
    slope: real(log_slope(s0, n, delta, lag)),
  };
  let at_strike = real(vol.implied_volatility(s0, strike, t_fix));
  // Hagan §3: the swaplet takes the at-the-money vol and the option out of the money its strike's.
  let adjustment = model.swaplet(real(vol.implied_volatility(s0, s0, t_fix)));
  let forward_value = match (is_caplet, k >= s) {
    // (3.5c) needs a positive forward; at or below 0 a Black rate keeps its side of 0, so the
    // adjusted forward takes the Black caplet's sign-preserving reading.
    _ if model.kind == VolatilityQuoteKind::Lognormal && (s.is_nan() || s <= 0.0) => {
      let black = if is_caplet {
        black_forward_caplet::<f64>
      } else {
        black_forward_floorlet::<f64>
      };
      black(s + adjustment, k, model.t, at_strike)
    }
    (true, true) => model.caplet(k, at_strike),
    (true, false) => s + adjustment - k + model.floorlet(k, at_strike),
    (false, true) if k > s => k - s - adjustment + model.caplet(k, at_strike),
    (false, _) => model.floorlet(k, at_strike),
  };
  CmsValuation {
    forward_swap_rate: s0,
    convexity_adjustment: T::from_f64_fast(adjustment),
    npv: notional * accrual * discount * T::from_f64_fast(forward_value),
  }
}

#[cfg(test)]
mod tests;
