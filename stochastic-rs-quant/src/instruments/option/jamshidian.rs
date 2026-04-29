//! Jamshidian decomposition for Hull-White European swaptions.
//!
//! $$
//! V_{\mathrm{payer}}(0)=\sum_{i=1}^{n} c_i\,\mathrm{ZBP}(0,T_0,T_i,X_i),\qquad
//! \sum_{i=1}^{n} c_i\,P(T_0,T_i;r^\star)=N
//! $$
//!
//! The cash-flow strip of the underlying receiver coupon bond is expressed as a
//! portfolio of zero-coupon-bond puts (payer swaption) or calls (receiver
//! swaption), each with a Jamshidian strike $X_i=A(T_0,T_i)e^{-B(T_0,T_i)r^\star}$.
//!
//! Reference: Jamshidian, "An Exact Bond Option Formula", Journal of Finance
//! 44(1), 205-209 (1989).
//!
//! Reference: Brigo & Mercurio, "Interest Rate Models — Theory and Practice",
//! Springer 2nd ed. (2006), §3.3.

use stochastic_rs_distributions::special::norm_cdf;

use super::types::SwaptionDirection;
use crate::curves::DiscountCurve;
use crate::traits::FloatExt;

/// Analytic Hull-White European swaption priced via Jamshidian decomposition.
///
/// * `coupon_times` and `accrual_factors` describe the underlying fixed leg
///   (time from today to each coupon payment and its accrual fraction). The
///   final coupon time is the swap maturity $T_n$; the notional redemption is
///   implicitly added at that date.
/// * `mean_reversion` is $a$, `sigma` is $\sigma$ in the Hull-White SDE.
/// * `curve` supplies the initial market discount factors $P^M(0,\cdot)$.
#[allow(clippy::too_many_arguments)]
pub fn price_jamshidian_hull_white<T: FloatExt>(
  direction: SwaptionDirection,
  strike: T,
  notional: T,
  expiry: T,
  coupon_times: &[T],
  accrual_factors: &[T],
  mean_reversion: T,
  sigma: T,
  curve: &DiscountCurve<T>,
) -> T {
  assert!(
    !coupon_times.is_empty(),
    "at least one coupon time is required"
  );
  assert_eq!(
    coupon_times.len(),
    accrual_factors.len(),
    "coupon_times and accrual_factors must have equal length"
  );
  assert!(expiry > T::zero(), "expiry must be strictly positive");
  for &t in coupon_times {
    assert!(t >= expiry, "coupon times must be on or after the expiry");
  }

  let a = mean_reversion;
  let t0 = expiry;
  let n = coupon_times.len();

  let cashflows: Vec<T> = (0..n)
    .map(|i| {
      let base = notional * strike * accrual_factors[i];
      if i == n - 1 { base + notional } else { base }
    })
    .collect();

  let b_fn = |t_from: T, t_to: T| -> T {
    let tau = t_to - t_from;
    if a.abs() < T::from_f64_fast(1e-14) {
      tau
    } else {
      (T::one() - (-a * tau).exp()) / a
    }
  };

  let f_market = |t: T| -> T {
    let eps = T::from_f64_fast(1e-4);
    let t_lo = if t > eps { t - eps } else { T::zero() };
    let t_hi = t + eps;
    curve.forward_rate(t_lo, t_hi)
  };

  let aa_fn = |t_from: T, t_to: T| -> T {
    let p_to = curve.discount_factor(t_to);
    let p_from = curve.discount_factor(t_from);
    if p_from <= T::zero() {
      return T::zero();
    }
    let b_val = b_fn(t_from, t_to);
    let f_val = f_market(t_from);
    let two_a = T::from_f64_fast(2.0) * a;
    let variance = if a.abs() < T::from_f64_fast(1e-14) {
      sigma * sigma * t_from * b_val * b_val
    } else {
      sigma * sigma / (T::from_f64_fast(4.0) * a)
        * (T::one() - (-two_a * t_from).exp())
        * b_val
        * b_val
    };
    (p_to / p_from) * (b_val * f_val - variance).exp()
  };

  let mut r_star = f_market(t0);
  for _ in 0..200 {
    let mut f_val = -notional;
    let mut fp_val = T::zero();
    for i in 0..n {
      let b_i = b_fn(t0, coupon_times[i]);
      let a_i = aa_fn(t0, coupon_times[i]);
      let v = a_i * (-b_i * r_star).exp();
      f_val += cashflows[i] * v;
      fp_val += cashflows[i] * (-b_i) * v;
    }
    if fp_val.abs() < T::from_f64_fast(1e-16) {
      break;
    }
    let dr = f_val / fp_val;
    r_star -= dr;
    if dr.abs() < T::from_f64_fast(1e-12) {
      break;
    }
  }

  let p0 = curve.discount_factor(t0);

  let mut total = T::zero();
  for i in 0..n {
    let t_i = coupon_times[i];
    let b_i = b_fn(t0, t_i);
    let a_i = aa_fn(t0, t_i);
    let x_i = a_i * (-b_i * r_star).exp();
    let p_ti = curve.discount_factor(t_i);

    let two_a = T::from_f64_fast(2.0) * a;
    let variance_p = if a.abs() < T::from_f64_fast(1e-14) {
      sigma * sigma * t0 * b_i * b_i
    } else {
      sigma * sigma * (T::one() - (-two_a * t0).exp()) / two_a * b_i * b_i
    };
    let sigma_p = variance_p.sqrt();

    let contribution = if sigma_p <= T::zero() || x_i <= T::zero() || p0 <= T::zero() {
      let put_intrinsic = (x_i * p0 - p_ti).max(T::zero());
      let call_intrinsic = (p_ti - x_i * p0).max(T::zero());
      match direction {
        SwaptionDirection::Payer => cashflows[i] * put_intrinsic,
        SwaptionDirection::Receiver => cashflows[i] * call_intrinsic,
      }
    } else {
      let sp = sigma_p.to_f64().unwrap_or(0.0);
      let ln_term = (p_ti / (x_i * p0)).to_f64().unwrap_or(0.0).ln();
      let h = ln_term / sp + 0.5 * sp;
      let xp0 = (x_i * p0).to_f64().unwrap_or(0.0);
      let pti = p_ti.to_f64().unwrap_or(0.0);
      match direction {
        SwaptionDirection::Payer => {
          let zbp = xp0 * norm_cdf(-h + sp) - pti * norm_cdf(-h);
          cashflows[i] * T::from_f64_fast(zbp)
        }
        SwaptionDirection::Receiver => {
          let zbc = pti * norm_cdf(h) - xp0 * norm_cdf(h - sp);
          cashflows[i] * T::from_f64_fast(zbc)
        }
      }
    };

    total += contribution;
  }

  total
}

/// Thin wrapper around [`price_jamshidian_hull_white`] that stores the swaption
/// description alongside the Hull-White parameters.
#[derive(Debug, Clone)]
pub struct JamshidianHullWhiteSwaption<T: FloatExt> {
  /// Payoff direction.
  pub direction: SwaptionDirection,
  /// Fixed strike $K$.
  pub strike: T,
  /// Swap notional.
  pub notional: T,
  /// Time from valuation to exercise (years).
  pub expiry: T,
  /// Fixed-leg coupon times $T_1 \le \dots \le T_n$ (years from valuation).
  pub coupon_times: Vec<T>,
  /// Fixed-leg accrual factors $\delta_i$.
  pub accrual_factors: Vec<T>,
  /// Hull-White mean reversion $a$.
  pub mean_reversion: T,
  /// Hull-White volatility $\sigma$.
  pub sigma: T,
}

impl<T: FloatExt> JamshidianHullWhiteSwaption<T> {
  /// Build a Jamshidian-priced European swaption under Hull-White.
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    direction: SwaptionDirection,
    strike: T,
    notional: T,
    expiry: T,
    coupon_times: Vec<T>,
    accrual_factors: Vec<T>,
    mean_reversion: T,
    sigma: T,
  ) -> Self {
    Self {
      direction,
      strike,
      notional,
      expiry,
      coupon_times,
      accrual_factors,
      mean_reversion,
      sigma,
    }
  }

  /// Present value given the initial discount curve.
  pub fn price(&self, curve: &DiscountCurve<T>) -> T {
    price_jamshidian_hull_white(
      self.direction,
      self.strike,
      self.notional,
      self.expiry,
      &self.coupon_times,
      &self.accrual_factors,
      self.mean_reversion,
      self.sigma,
      curve,
    )
  }
}
