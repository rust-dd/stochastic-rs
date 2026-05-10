//! # Hull-White (1990) extended Vasicek short-rate model — closed-form ZCB.
//!
//! $$
//! dr_t = (\theta(t) - a\,r_t)\,dt + \sigma\,dW_t
//! $$
//!
//! Hull-White is an affine short-rate model belonging to the
//! `P(t,T) = exp(A(t,T) - r(t)·B(t,T))` family with
//!
//! $$
//! B(t,T) = \frac{1 - e^{-a(T-t)}}{a}
//! $$
//!
//! When the deterministic mean-reversion target $\theta(t)$ is calibrated to
//! match an initial market discount curve $P^M(0,\cdot)$ (the standard
//! no-arbitrage extension; see Brigo & Mercurio 2007 §3.3.2), the closed form
//! collapses to
//!
//! $$
//! P(t,T) = \frac{P^M(0,T)}{P^M(0,t)}\,
//!          \exp\!\Big(B(t,T)\,f^M(0,t)
//!                     - \frac{\sigma^2}{4a}(1 - e^{-2at})\,B(t,T)^2
//!                     - B(t,T)\,r(t)\Big)
//! $$
//!
//! where $f^M(0,t) := -\partial_T \ln P^M(0,t)$ is the initial instantaneous
//! forward rate. This form is **deterministic** (no `Utc::now()` poisoning),
//! reduces to $P(0,T) = P^M(0,T)$ when $t=0$ and $r(0) = f^M(0,0)$, and works
//! for any positive $a$, $\sigma$.
//!
//! References:
//! - Hull & White (1990), "Pricing Interest Rate Derivative Securities",
//!   Review of Financial Studies, 3(4), 573-592.
//! - Brigo & Mercurio (2007), §3.3.2.
//! - Kisbye & Meier (2017), "Consistency of extended Nelson-Siegel curve
//!   families with the Ho-Lee and Hull and White short rate models",
//!   arXiv:1707.02496, §3 (eq. 3.3-3.4 + θ(t) Brigo formula).
use crate::curves::DiscountCurve;
use crate::traits::PricerExt;
use crate::traits::TimeExt;

/// Hull-White ZCB pricer parameterised by the calibrated initial discount
/// curve. The struct stores the *projection* of that curve onto the two
/// time points the closed form actually needs ($P^M(0,t)$, $P^M(0,T)$,
/// $f^M(0,t)$), which keeps the type Python-bindable (no `&'a` lifetime,
/// no `fn(f64)->f64` field) and decouples pricing from curve interpolation.
///
/// Use [`HullWhite::from_curve`] to build the struct from a [`DiscountCurve`];
/// or set the discount/forward fields directly if you already have them.
#[derive(Default, Debug, Clone)]
pub struct HullWhite {
  /// Current short rate $r(t)$.
  pub r_t: f64,
  /// Mean-reversion speed $a > 0$.
  pub alpha: f64,
  /// Short-rate volatility $\sigma > 0$.
  pub sigma: f64,
  /// Time to maturity $\tau = T - t$ in years.
  pub tau: f64,
  /// Evaluation time $t$ measured from the curve's inception in years
  /// (i.e., the curve was calibrated at "time 0", and `t` is how far in
  /// the future we are pricing). For pricing as-of curve inception, use `t = 0`.
  pub t: f64,
  /// Initial market discount factor at evaluation time, $P^M(0, t)$.
  pub p0_at_t: f64,
  /// Initial market discount factor at maturity, $P^M(0, T) = P^M(0, t+\tau)$.
  pub p0_at_maturity: f64,
  /// Initial market instantaneous forward rate at evaluation time, $f^M(0, t)$.
  pub f0_at_t: f64,
  /// Optional evaluation date (used by [`TimeExt`] for date-aware downstream).
  pub eval: Option<chrono::NaiveDate>,
  /// Optional maturity date (used by [`TimeExt`] for date-aware downstream).
  pub expiration: Option<chrono::NaiveDate>,
}

impl HullWhite {
  /// Build a Hull-White pricer by projecting a calibrated [`DiscountCurve`]
  /// onto the time points the closed form needs.
  ///
  /// The instantaneous forward $f^M(0, t) = -\partial_T \ln P^M(0, t)$ is
  /// estimated by a centered finite difference on the curve's log-discount
  /// factor. The bump width is chosen as `max(1e-4, 1e-4·t)` — small enough
  /// to keep the discretisation error well below the bond's typical scale,
  /// large enough to avoid catastrophic cancellation.
  pub fn from_curve(
    curve: &DiscountCurve<f64>,
    r_t: f64,
    alpha: f64,
    sigma: f64,
    t: f64,
    tau: f64,
    eval: Option<chrono::NaiveDate>,
    expiration: Option<chrono::NaiveDate>,
  ) -> Self {
    let p0_at_t = curve.discount_factor(t);
    let p0_at_maturity = curve.discount_factor(t + tau);
    let f0_at_t = instantaneous_forward(curve, t);
    Self {
      r_t,
      alpha,
      sigma,
      tau,
      t,
      p0_at_t,
      p0_at_maturity,
      f0_at_t,
      eval,
      expiration,
    }
  }
}

/// Centered finite-difference estimate of $f^M(0, t) = -\partial_T \ln P^M(0, t)$.
///
/// **Curve anchoring:** The estimate uses the curve's `forward_rate(t1, t2)`
/// helper, which is `-(ln D(t2) - ln D(t1))/(t2-t1)`. For accurate near-spot
/// pricing the calibrated [`DiscountCurve`] must include `t = 0` (with
/// `D(0) = 1`); otherwise log-linear interpolation is constant in
/// `(0, points[0].time]` and the forward extraction blows up there.
fn instantaneous_forward(curve: &DiscountCurve<f64>, t: f64) -> f64 {
  let h = (1e-4_f64).max(1e-4 * t);
  if t > h {
    curve.forward_rate(t - h, t + h)
  } else {
    // Near t=0 the centered stencil straddles the curve origin.
    curve.forward_rate(0.0, h)
  }
}

impl PricerExt for HullWhite {
  fn calculate_call_put(&self) -> (f64, f64) {
    let price = self.calculate_price();
    (price, price)
  }

  fn calculate_price(&self) -> f64 {
    let a = self.alpha;
    let sigma = self.sigma;
    let r = self.r_t;
    let tau = self.tau;
    let t = self.t;

    // B(t,T) = (1 - e^{-a τ}) / a
    let b = (1.0 - (-a * tau).exp()) / a;

    // Exponent: B·f^M(0,t) - σ²/(4a)·(1-e^{-2at})·B² - B·r(t)
    let exponent =
      b * self.f0_at_t - (sigma * sigma) / (4.0 * a) * (1.0 - (-2.0 * a * t).exp()) * b * b - b * r;

    self.p0_at_maturity / self.p0_at_t * exponent.exp()
  }
}

impl TimeExt for HullWhite {
  fn tau(&self) -> Option<f64> {
    Some(self.tau)
  }

  fn eval(&self) -> Option<chrono::NaiveDate> {
    self.eval
  }

  fn expiration(&self) -> Option<chrono::NaiveDate> {
    self.expiration
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::curves::types::CurvePoint;
  use crate::curves::types::InterpolationMethod;

  fn flat_curve(rate: f64) -> DiscountCurve<f64> {
    // Anchor t=0 explicitly with D(0)=1 so log-linear interpolation gives the
    // correct flat-rate forward in the [0, 0.5) region.
    let mut pts = vec![CurvePoint {
      time: 0.0,
      discount_factor: 1.0,
    }];
    for i in 1..=20 {
      let t = i as f64 * 0.5;
      pts.push(CurvePoint {
        time: t,
        discount_factor: (-rate * t).exp(),
      });
    }
    DiscountCurve::new(pts, InterpolationMethod::LogLinearOnDiscountFactors)
  }

  #[test]
  fn zcb_at_zero_tau_equals_one() {
    let curve = flat_curve(0.05);
    let h = HullWhite::from_curve(&curve, 0.05, 0.5, 0.01, 0.5, 0.0, None, None);
    let p = h.calculate_price();
    assert!((p - 1.0).abs() < 1e-12, "P(t,t) must be 1, got {p}");
  }

  #[test]
  fn zcb_at_t_zero_matches_market_curve() {
    // No-arbitrage property of Hull-White: at t = 0 with r(0) = f^M(0, 0),
    // the closed-form ZCB must reproduce the market discount factor exactly.
    let rate = 0.05;
    let curve = flat_curve(rate);
    // For a flat curve at rate r_market, f^M(0, 0) = r_market.
    let f0_at_zero = instantaneous_forward(&curve, 0.0);
    let h = HullWhite::from_curve(&curve, f0_at_zero, 0.5, 0.01, 0.0, 2.0, None, None);
    let p_hw = h.calculate_price();
    let p_market = curve.discount_factor(2.0);
    assert!(
      (p_hw - p_market).abs() < 1e-6,
      "HW@t=0 must match market: hw={p_hw} market={p_market}"
    );
  }

  #[test]
  fn zcb_finite_and_positive() {
    let curve = flat_curve(0.05);
    let h = HullWhite::from_curve(&curve, 0.05, 0.5, 0.01, 0.0, 2.0, None, None);
    let p = h.calculate_price();
    assert!(
      p.is_finite() && p > 0.0,
      "ZCB must be finite-positive, got {p}"
    );
  }

  #[test]
  fn zcb_decreases_with_short_rate() {
    let curve = flat_curve(0.05);
    let make = |r| HullWhite::from_curve(&curve, r, 0.5, 0.01, 0.0, 1.0, None, None);
    let p_low = make(0.02).calculate_price();
    let p_high = make(0.08).calculate_price();
    assert!(
      p_high < p_low,
      "ZCB must decrease with short rate: p(0.02)={p_low} vs p(0.08)={p_high}"
    );
  }

  #[test]
  fn zcb_below_one_for_positive_rate_and_tau() {
    let curve = flat_curve(0.05);
    let h = HullWhite::from_curve(&curve, 0.05, 0.5, 0.01, 0.0, 5.0, None, None);
    let p = h.calculate_price();
    assert!(p < 1.0 && p > 0.0, "ZCB out of range: {p}");
  }

  #[test]
  fn deterministic_no_clock_dependency() {
    // The fix removes Utc::now()-dependence: prices computed at any wall-clock
    // moment must be equal for identical inputs.
    let curve = flat_curve(0.04);
    let make = || HullWhite::from_curve(&curve, 0.04, 0.3, 0.015, 1.0, 2.0, None, None);
    let p1 = make().calculate_price();
    let p2 = make().calculate_price();
    assert_eq!(
      p1.to_bits(),
      p2.to_bits(),
      "HW must be deterministic, got {p1} != {p2}"
    );
  }
}
