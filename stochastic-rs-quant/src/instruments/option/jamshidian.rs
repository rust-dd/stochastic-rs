//! Jamshidian decomposition of Hull-White European swaptions:
//!
//! $$
//! V_{\mathrm{payer}}(0)=\sum_{i=1}^{n} c_i\,\mathrm{ZBP}(0,T_0,T_i,X_i),\qquad
//! \sum_{i=1}^{n} c_i\,P(T_0,T_i;r^\star)=N
//! $$
//!
//! with zero-coupon-bond puts (payer) or calls (receiver) at `X_i = A(T₀, T_i)e^{−B(T₀, T_i)r*}`.
//! Reference: Jamshidian (1989), J. Finance 44(1), 205–209
//! Reference: Brigo & Mercurio (2006), 2nd ed., §3.3, DOI 10.1007/978-3-540-34604-3

use stochastic_rs_distributions::special::norm_cdf;

use super::types::SwaptionDirection;
use crate::curves::DiscountCurve;
use crate::traits::RealExt;

/// Hull-White swaption by Jamshidian, the notional repaid with the last coupon; at `tau = 0` the
/// swap's value floored at 0, NaN for a negative or NaN `tau` or no root `r*`; panics on a bad leg.
#[allow(clippy::too_many_arguments)]
pub fn price_jamshidian_hull_white<T: RealExt>(
  direction: SwaptionDirection,
  strike: T,
  notional: T,
  tau: T,
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
  if tau.is_nan() || tau < T::zero() {
    return T::nan();
  }
  for &t in coupon_times {
    assert!(t >= tau, "coupon times must be on or after `tau`");
  }

  let a = mean_reversion;
  let t0 = tau;
  let n = coupon_times.len();

  let cashflows: Vec<T> = (0..n)
    .map(|i| {
      let base = notional * strike * accrual_factors[i];
      if i == n - 1 { base + notional } else { base }
    })
    .collect();

  if tau == T::zero() {
    let mut bond = T::zero();
    for (cashflow, t) in cashflows.iter().zip(coupon_times) {
      bond += *cashflow * curve.discount_factor(*t);
    }
    let swap = match direction {
      SwaptionDirection::Payer => notional - bond,
      SwaptionDirection::Receiver => bond - notional,
    };
    return swap.max_or_nan(T::zero());
  }

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
      return T::nan();
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
  let mut solved = false;
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
    let dr = f_val / fp_val;
    if f_val == T::zero() || !dr.is_finite() {
      solved = f_val == T::zero();
      break;
    }
    r_star -= dr;
    if dr.abs() < T::from_f64_fast(1e-12) {
      solved = true;
      break;
    }
  }
  if !solved {
    return T::nan();
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

    if !(x_i > T::zero() && p0 > T::zero()) {
      return T::nan();
    }
    let contribution = if sigma_p <= T::zero() {
      let put_intrinsic = (x_i * p0 - p_ti).max_or_nan(T::zero());
      let call_intrinsic = (p_ti - x_i * p0).max_or_nan(T::zero());
      match direction {
        SwaptionDirection::Payer => cashflows[i] * put_intrinsic,
        SwaptionDirection::Receiver => cashflows[i] * call_intrinsic,
      }
    } else {
      let sp = sigma_p.to_f64().unwrap_or(f64::NAN);
      let ln_term = (p_ti / (x_i * p0)).to_f64().unwrap_or(f64::NAN).ln();
      let h = ln_term / sp + 0.5 * sp;
      let xp0 = (x_i * p0).to_f64().unwrap_or(f64::NAN);
      let pti = p_ti.to_f64().unwrap_or(f64::NAN);
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
pub struct JamshidianHullWhiteSwaption<T: RealExt> {
  /// Payoff direction.
  pub direction: SwaptionDirection,
  /// Fixed strike $K$.
  pub strike: T,
  /// Swap notional.
  pub notional: T,
  /// Time from valuation to exercise (years).
  pub tau: T,
  /// Fixed-leg coupon times $T_1 \le \dots \le T_n$ (years from valuation).
  pub coupon_times: Vec<T>,
  /// Fixed-leg accrual factors $\delta_i$.
  pub accrual_factors: Vec<T>,
  /// Hull-White mean reversion $a$.
  pub mean_reversion: T,
  /// Hull-White volatility $\sigma$.
  pub sigma: T,
}

impl<T: RealExt> JamshidianHullWhiteSwaption<T> {
  /// Panics unless `mean_reversion` is finite and `sigma` finite and not negative.
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    direction: SwaptionDirection,
    strike: T,
    notional: T,
    tau: T,
    coupon_times: Vec<T>,
    accrual_factors: Vec<T>,
    mean_reversion: T,
    sigma: T,
  ) -> Self {
    assert!(
      mean_reversion.is_finite(),
      "mean_reversion must satisfy `mean_reversion.is_finite()`, got mean_reversion = \
       {mean_reversion:?}"
    );
    assert!(
      sigma.is_finite() && sigma >= T::zero(),
      "sigma must satisfy `sigma.is_finite() && sigma >= T::zero()`, got sigma = {sigma:?}"
    );
    Self {
      direction,
      strike,
      notional,
      tau,
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
      self.tau,
      &self.coupon_times,
      &self.accrual_factors,
      self.mean_reversion,
      self.sigma,
      curve,
    )
  }
}
