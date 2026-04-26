//! Multi-curve framework for post-crisis interest rate modeling.
//!
//! Reference: Bianchetti, "Two Curves, One Price", arXiv:0905.2770 (2009).
//!
//! Reference: Pallavicini & Tarenghi, "Interest-Rate Modeling with Multiple Yield Curves",
//! arXiv:1006.4767 (2010).
//!
//! Reference: Cuchiero, Fontana & Gnoatto, "Affine multiple yield curve models",
//! arXiv:1603.00527 (2016).
//!
//! Post-crisis, a single yield curve is insufficient. The framework uses:
//! - **Discount curve**: built from OIS/SOFR rates, used for present value computation.
//! - **Forecast curves**: one per floating rate tenor, used for projecting future fixings.
//!
//! $$
//! \text{PV}_{\text{float}} = \sum_j D_{\text{OIS}}(t_j)\,\delta_j\,L_{\text{forecast}}(t_{j-1}, t_j)\,N
//! $$

use ndarray::Array1;

use super::discount_curve::DiscountCurve;
use crate::traits::FloatExt;

/// Multi-curve framework holding a discount curve and tenor-specific forecast curves.
#[derive(Debug, Clone)]
pub struct MultiCurve<T: FloatExt> {
  /// OIS / SOFR discount curve used for present value computation.
  pub discount: DiscountCurve<T>,
  /// Forecast curves keyed by tenor label (e.g., "1M", "3M", "6M").
  pub forecasts: Vec<(String, DiscountCurve<T>)>,
}

impl<T: FloatExt> MultiCurve<T> {
  /// Create a multi-curve framework with a discount curve.
  pub fn new(discount: DiscountCurve<T>) -> Self {
    Self {
      discount,
      forecasts: Vec::new(),
    }
  }

  /// Add a forecast curve for a given tenor.
  pub fn add_forecast(&mut self, tenor: impl Into<String>, curve: DiscountCurve<T>) {
    self.forecasts.push((tenor.into(), curve));
  }

  /// Get the forecast curve for a given tenor.
  pub fn forecast(&self, tenor: &str) -> Option<&DiscountCurve<T>> {
    self
      .forecasts
      .iter()
      .find(|(t, _)| t == tenor)
      .map(|(_, c)| c)
  }

  /// Projected simple forward rate from the forecast curve for `tenor` over `[t1, t2]`.
  ///
  /// $$
  /// L(t_1, t_2) = \frac{1}{\delta}\left(\frac{D_{\text{forecast}}(t_1)}{D_{\text{forecast}}(t_2)} - 1\right)
  /// $$
  pub fn projected_forward(&self, tenor: &str, t1: T, t2: T) -> Option<T> {
    self
      .forecast(tenor)
      .map(|fc| fc.simple_forward_rate(t1, t2))
  }

  /// Basis spread between a forecast curve and the OIS discount curve.
  ///
  /// $$
  /// s(t_1, t_2) = L_{\text{forecast}}(t_1, t_2) - L_{\text{OIS}}(t_1, t_2)
  /// $$
  pub fn basis_spread(&self, tenor: &str, t1: T, t2: T) -> Option<T> {
    let fwd_forecast = self.projected_forward(tenor, t1, t2)?;
    let fwd_ois = self.discount.simple_forward_rate(t1, t2);
    Some(fwd_forecast - fwd_ois)
  }

  /// Present value of a floating leg using OIS discounting and forecast curve projection.
  ///
  /// Payment schedule: $t_0, t_1, \ldots, t_n$ with $\delta_i = t_i - t_{i-1}$.
  pub fn pv_floating_leg(&self, tenor: &str, schedule: &Array1<T>, notional: T) -> Option<T> {
    let fc = self.forecast(tenor)?;
    let n = schedule.len();
    if n < 2 {
      return Some(T::zero());
    }

    let mut pv = T::zero();
    for i in 1..n {
      let t0 = schedule[i - 1];
      let t1 = schedule[i];
      let delta = t1 - t0;
      let fwd = fc.simple_forward_rate(t0, t1);
      let df = self.discount.discount_factor(t1);
      pv += df * delta * fwd * notional;
    }
    Some(pv)
  }

  /// Present value of a fixed leg using OIS discounting.
  pub fn pv_fixed_leg(&self, schedule: &Array1<T>, fixed_rate: T, notional: T) -> T {
    let n = schedule.len();
    if n < 2 {
      return T::zero();
    }

    let mut pv = T::zero();
    for i in 1..n {
      let t0 = schedule[i - 1];
      let t1 = schedule[i];
      let delta = t1 - t0;
      let df = self.discount.discount_factor(t1);
      pv += df * delta * fixed_rate * notional;
    }
    pv
  }

  /// Fair swap rate under the multi-curve framework.
  ///
  /// The fair rate $S$ satisfies $\text{PV}_{\text{fixed}} = \text{PV}_{\text{float}}$:
  /// $$
  /// S = \frac{\sum_j D_{\text{OIS}}(t_j)\,\delta_j\,L_{\text{forecast}}(t_{j-1}, t_j)}
  ///          {\sum_j D_{\text{OIS}}(t_j)\,\delta_j}
  /// $$
  pub fn fair_swap_rate(&self, tenor: &str, schedule: &Array1<T>) -> Option<T> {
    let fc = self.forecast(tenor)?;
    let n = schedule.len();
    if n < 2 {
      return Some(T::zero());
    }

    let mut float_leg = T::zero();
    let mut annuity = T::zero();

    for i in 1..n {
      let t0 = schedule[i - 1];
      let t1 = schedule[i];
      let delta = t1 - t0;
      let df = self.discount.discount_factor(t1);
      let fwd = fc.simple_forward_rate(t0, t1);
      float_leg += df * delta * fwd;
      annuity += df * delta;
    }

    Some(float_leg / annuity)
  }
}

#[cfg(test)]
mod tests {
  use super::super::types::CurvePoint;
  use super::super::types::InterpolationMethod;
  use super::*;

  fn flat(rate: f64) -> DiscountCurve<f64> {
    DiscountCurve::new(
      vec![
        CurvePoint { time: 0.5, discount_factor: (-rate * 0.5).exp() },
        CurvePoint { time: 1.0, discount_factor: (-rate * 1.0).exp() },
        CurvePoint { time: 2.0, discount_factor: (-rate * 2.0).exp() },
      ],
      InterpolationMethod::LogLinearOnDiscountFactors,
    )
  }

  #[test]
  fn multi_curve_stores_forecasts() {
    let mut mc = MultiCurve::new(flat(0.04));
    mc.add_forecast("3M", flat(0.045));
    assert!(mc.forecast("3M").is_some());
    assert!(mc.forecast("6M").is_none());
  }

  #[test]
  fn basis_spread_zero_for_identical_curves() {
    let mut mc = MultiCurve::new(flat(0.04));
    mc.add_forecast("3M", flat(0.04));
    let spread = mc.basis_spread("3M", 0.5, 1.0).unwrap();
    assert!(spread.abs() < 1e-9, "spread should be zero for identical curves: {spread}");
  }
}
