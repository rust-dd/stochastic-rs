//! Hull–White extended Vasicek zero-coupon bond with $\theta(t)$ fitted to an initial market
//! curve $P^M(0,\cdot)$, where $f^M(0,t) = -\partial_T \ln P^M(0,t)$.
//!
//! $$
//! dr_t = (\theta(t) - a\,r_t)\,dt + \sigma\,dW_t,\qquad
//! B(t,T) = \frac{1 - e^{-a(T-t)}}{a}
//! $$
//!
//! $$
//! P(t,T) = \frac{P^M(0,T)}{P^M(0,t)}\,
//!          \exp\!\Big(B(t,T)\,f^M(0,t)
//!                     - \frac{\sigma^2}{4a}(1 - e^{-2at})\,B(t,T)^2
//!                     - B(t,T)\,r(t)\Big)
//! $$
//!
//! Reference: Hull & White (1990), Review of Financial Studies 3(4), 573–592
//! Reference: Brigo & Mercurio (2007), "Interest Rate Models: Theory and Practice", §3.3.2
//! Reference: Kisbye & Meier (2017), arXiv:1707.02496, §3, eq. 3.3–3.4
use crate::curves::DiscountCurve;
use crate::traits::ShortRatePricer;

/// Hull–White model with the market curve projected onto $P^M(0,t)$, $P^M(0,T)$ and $f^M(0,t)$.
/// Built by [`Self::from_curve`] it prices any `tau`; built from fields, only `pinned_tau`.
#[derive(Default, Debug, Clone)]
pub struct HullWhite {
  /// Mean-reversion speed $a > 0$.
  pub alpha: f64,
  /// Short-rate volatility $\sigma > 0$.
  pub sigma: f64,
  /// Evaluation time $t$ in years from the curve's inception; `0` prices as of inception.
  pub t: f64,
  /// Initial market discount factor at evaluation time, $P^M(0, t)$.
  pub p0_at_t: f64,
  /// $P^M(0, t+\tau)$ at `pinned_tau`; read only when `curve` is `None`.
  pub p0_at_maturity: f64,
  /// Initial market instantaneous forward rate at evaluation time, $f^M(0, t)$.
  pub f0_at_t: f64,
  /// The horizon `p0_at_maturity` was projected for; without a `curve`, a query at another `tau`
  /// panics.
  pub pinned_tau: f64,
  /// The curve [`from_curve`](Self::from_curve) keeps, so each query reads $P^M(0, t+\tau)$ from
  /// it; `None` for a model built from fields, which prices only at `pinned_tau`.
  pub curve: Option<DiscountCurve<f64>>,
}

impl HullWhite {
  /// Projects `curve` at `t` and `t + tau` and keeps it, so the model prices at any `tau`;
  /// $f^M(0, t)$ is a centred difference of the log discount factor of width `max(1e-4, 1e-4·t)`.
  pub fn from_curve(curve: &DiscountCurve<f64>, alpha: f64, sigma: f64, t: f64, tau: f64) -> Self {
    let p0_at_t = curve.discount_factor(t);
    let p0_at_maturity = curve.discount_factor(t + tau);
    let f0_at_t = instantaneous_forward(curve, t);
    Self {
      alpha,
      sigma,
      t,
      p0_at_t,
      p0_at_maturity,
      f0_at_t,
      pinned_tau: tau,
      curve: Some(curve.clone()),
    }
  }
}

/// Centred finite-difference estimate of $f^M(0, t)$. The curve needs a pillar at `t = 0`:
/// log-linear interpolation is flat on `(0, points[0].time]` and the forward degenerates there.
fn instantaneous_forward(curve: &DiscountCurve<f64>, t: f64) -> f64 {
  let h = (1e-4_f64).max(1e-4 * t);
  if t > h {
    curve.forward_rate(t - h, t + h)
  } else {
    // Near t=0 the centered stencil straddles the curve origin.
    curve.forward_rate(0.0, h)
  }
}

impl ShortRatePricer for HullWhite {
  /// Reads $P^M(0, t+\tau)$ from `curve` when it is set and from `p0_at_maturity` otherwise;
  /// panics unless `tau ≥ 0` and, without a curve, `tau == pinned_tau`.
  fn zero_coupon_price(&self, r0: f64, tau: f64) -> f64 {
    assert!(tau >= 0.0, "tau must be non-negative (got {tau})");
    let a = self.alpha;
    let sigma = self.sigma;
    let t = self.t;

    let p0_at_maturity = match &self.curve {
      Some(curve) => curve.discount_factor(t + tau),
      None => {
        let pinned_tau = self.pinned_tau;
        assert!(
          tau == pinned_tau,
          "tau must equal the maturity this HullWhite was built for when no \
           curve is stored (got tau={tau}, built for tau={pinned_tau})"
        );
        self.p0_at_maturity
      }
    };

    // B(t,T) = (1 - e^{-a τ}) / a
    let b = (1.0 - (-a * tau).exp()) / a;

    // Exponent: B·f^M(0,t) - σ²/(4a)·(1-e^{-2at})·B² - B·r(t)
    let exponent = b * self.f0_at_t
      - (sigma * sigma) / (4.0 * a) * (1.0 - (-2.0 * a * t).exp()) * b * b
      - b * r0;

    p0_at_maturity / self.p0_at_t * exponent.exp()
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
    let h = HullWhite::from_curve(&curve, 0.5, 0.01, 0.5, 0.0);
    let p = h.zero_coupon_price(0.05, 0.0);
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
    let h = HullWhite::from_curve(&curve, 0.5, 0.01, 0.0, 2.0);
    let p_hw = h.zero_coupon_price(f0_at_zero, 2.0);
    let p_market = curve.discount_factor(2.0);
    assert!(
      (p_hw - p_market).abs() < 1e-6,
      "HW@t=0 must match market: hw={p_hw} market={p_market}"
    );
  }

  #[test]
  fn zcb_finite_and_positive() {
    let curve = flat_curve(0.05);
    let h = HullWhite::from_curve(&curve, 0.5, 0.01, 0.0, 2.0);
    let p = h.zero_coupon_price(0.05, 2.0);
    assert!(
      p.is_finite() && p > 0.0,
      "ZCB must be finite-positive, got {p}"
    );
  }

  #[test]
  fn zcb_decreases_with_short_rate() {
    let curve = flat_curve(0.05);
    let h = HullWhite::from_curve(&curve, 0.5, 0.01, 0.0, 1.0);
    let p_low = h.zero_coupon_price(0.02, 1.0);
    let p_high = h.zero_coupon_price(0.08, 1.0);
    assert!(
      p_high < p_low,
      "ZCB must decrease with short rate: p(0.02)={p_low} vs p(0.08)={p_high}"
    );
  }

  #[test]
  fn zcb_below_one_for_positive_rate_and_tau() {
    let curve = flat_curve(0.05);
    let h = HullWhite::from_curve(&curve, 0.5, 0.01, 0.0, 5.0);
    let p = h.zero_coupon_price(0.05, 5.0);
    assert!(p < 1.0 && p > 0.0, "ZCB out of range: {p}");
  }

  #[test]
  fn deterministic_no_clock_dependency() {
    // Identical inputs price identically at any wall-clock moment.
    let curve = flat_curve(0.04);
    let make = || HullWhite::from_curve(&curve, 0.3, 0.015, 1.0, 2.0);
    let p1 = make().zero_coupon_price(0.04, 2.0);
    let p2 = make().zero_coupon_price(0.04, 2.0);
    assert_eq!(
      p1.to_bits(),
      p2.to_bits(),
      "HW must be deterministic, got {p1} != {p2}"
    );
  }

  #[test]
  #[should_panic(expected = "tau must be non-negative")]
  fn zero_coupon_price_panics_on_negative_tau() {
    let curve = flat_curve(0.05);
    let h = HullWhite::from_curve(&curve, 0.5, 0.01, 0.0, 1.0);
    h.zero_coupon_price(0.05, -1.0);
  }

  /// A model built from a curve at `tau = 5` prices `tau = 1` bit-identically to one built at
  /// `tau = 1`: the curve is reprojected per query, never baked in at construction.
  #[test]
  fn zero_coupon_price_reprices_any_maturity_from_one_model() {
    let curve = flat_curve(0.05);
    let built_for_five = HullWhite::from_curve(&curve, 0.5, 0.01, 0.5, 5.0);
    let built_for_one = HullWhite::from_curve(&curve, 0.5, 0.01, 0.5, 1.0);
    let p_reused = built_for_five.zero_coupon_price(0.08, 1.0);
    let p_direct = built_for_one.zero_coupon_price(0.08, 1.0);
    assert_eq!(
      p_reused.to_bits(),
      p_direct.to_bits(),
      "a model built for tau=5.0, queried at tau=1.0, must match a model \
       built directly for tau=1.0: {p_reused} vs {p_direct}"
    );
  }

  /// Two curve-backed models built at different `tau`s agree at every shared query point.
  #[test]
  fn zero_coupon_price_is_independent_of_construction_tau() {
    let curve = flat_curve(0.05);
    let h = HullWhite::from_curve(&curve, 0.5, 0.01, 0.5, 2.0);
    for &tau in &[0.5, 1.0, 2.0, 3.0, 7.5] {
      let p = h.zero_coupon_price(0.08, tau);
      let p_from_scratch =
        HullWhite::from_curve(&curve, 0.5, 0.01, 0.5, tau).zero_coupon_price(0.08, tau);
      assert_eq!(
        p.to_bits(),
        p_from_scratch.to_bits(),
        "tau={tau}: reused model gave {p}, fresh model gave {p_from_scratch}"
      );
    }
  }

  #[test]
  #[should_panic(expected = "tau must equal the maturity this HullWhite was built for")]
  fn zero_coupon_price_panics_on_tau_mismatch_without_curve() {
    // Mirrors PyHullWhiteBond::new's "already have the projected numbers"
    // path, which has no curve to requery.
    let h = HullWhite {
      alpha: 0.5,
      sigma: 0.01,
      t: 0.0,
      p0_at_t: 1.0,
      p0_at_maturity: 0.9,
      f0_at_t: 0.05,
      pinned_tau: 1.0,
      curve: None,
    };
    h.zero_coupon_price(0.05, 2.0);
  }

  #[test]
  fn zero_coupon_price_without_curve_matches_curve_backed_at_pinned_tau() {
    let curve = flat_curve(0.05);
    let with_curve = HullWhite::from_curve(&curve, 0.5, 0.01, 0.0, 2.0);
    let raw = HullWhite {
      alpha: 0.5,
      sigma: 0.01,
      t: 0.0,
      p0_at_t: with_curve.p0_at_t,
      p0_at_maturity: with_curve.p0_at_maturity,
      f0_at_t: with_curve.f0_at_t,
      pinned_tau: 2.0,
      curve: None,
    };
    let p_with_curve = with_curve.zero_coupon_price(0.05, 2.0);
    let p_raw = raw.zero_coupon_price(0.05, 2.0);
    assert_eq!(
      p_with_curve.to_bits(),
      p_raw.to_bits(),
      "no-curve fallback must match the curve-backed computation at pinned_tau"
    );
  }
}
