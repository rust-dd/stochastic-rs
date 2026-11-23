//! Gaussian kernel density estimate with Silverman's rule-of-thumb bandwidth.
//!
//! $$
//! \hat f_h(x)=\frac{1}{nh}\sum_{i=1}^n\phi\!\left(\frac{x-x_i}{h}\right),\qquad
//! h=0.9\,\min\!\left(\hat\sigma,\frac{\mathrm{IQR}}{1.34}\right)n^{-1/5}
//! $$
//!
//! Reference: Silverman (1986), Density Estimation, §3.4.2, DOI: 10.1201/9781315140919

use std::f64::consts::PI;

use ndarray::Array1;
use ndarray::ArrayView1;

use crate::StatsError;
use crate::input::as_slice_or_copy;
use crate::input::reject_constant;
use crate::input::validate;
use crate::input::validate_view;
use crate::stationarity::common::unit_scaled;
use crate::traits::RealExt;

/// Gaussian KDE over a copy of the sample.
#[derive(Debug, Clone)]
pub struct GaussianKde {
  data: Array1<f64>,
  bandwidth: f64,
}

impl GaussianKde {
  /// `Err` on an empty or non-finite sample; panics unless `bandwidth` is finite and positive.
  ///
  /// ```
  /// use ndarray::array;
  /// use stochastic_rs_stats::kde::GaussianKde;
  ///
  /// let kde = GaussianKde::new(array![1.0, 2.0, 3.0].view(), 0.5).unwrap();
  /// assert!(kde.evaluate(1.5) > 0.0);
  /// ```
  pub fn new(data: ArrayView1<f64>, bandwidth: f64) -> Result<Self, StatsError> {
    assert!(
      bandwidth.is_finite(),
      "bandwidth must satisfy `bandwidth.is_finite()`, got bandwidth = {bandwidth:?}"
    );
    assert!(
      bandwidth > 0.0,
      "bandwidth must satisfy `bandwidth > 0.0`, got bandwidth = {bandwidth:?}"
    );
    validate_view(data, 1)?;
    Ok(Self {
      data: data.to_owned(),
      bandwidth,
    })
  }

  /// The sample with [`silverman_bandwidth`]'s bandwidth, and its `Err` cases.
  pub fn with_silverman_bandwidth(data: ArrayView1<f64>) -> Result<Self, StatsError> {
    let bandwidth = silverman_bandwidth(data)?;
    Self::new(data, bandwidth)
  }

  pub fn bandwidth(&self) -> f64 {
    self.bandwidth
  }

  pub fn evaluate(&self, x: f64) -> f64 {
    let norm = 1.0 / (self.bandwidth * (2.0 * PI).sqrt());
    let sum: f64 = self
      .data
      .iter()
      .map(|&xi| norm * (-0.5 * ((x - xi) / self.bandwidth).powi(2)).exp())
      .sum();
    sum / (self.data.len() as f64)
  }

  /// ```
  /// use ndarray::Array1;
  /// use ndarray::array;
  /// use stochastic_rs_stats::kde::GaussianKde;
  ///
  /// let kde = GaussianKde::new(array![1.0, 2.0, 3.0].view(), 0.5).unwrap();
  /// let ys = kde.evaluate_array(Array1::linspace(0.0, 5.0, 50).view());
  /// assert_eq!(ys.len(), 50);
  /// ```
  pub fn evaluate_array(&self, x: ArrayView1<f64>) -> Array1<f64> {
    x.mapv(|v| self.evaluate(v))
  }
}

/// Silverman's rule, `σ̂` alone when the interquartile range is zero; `Err` below two points, on a
/// non-finite one or zero spread, `Err(Numerical { stage: "silverman bandwidth" })` on overflow.
pub fn silverman_bandwidth(data: ArrayView1<f64>) -> Result<f64, StatsError> {
  let sample = as_slice_or_copy(data);
  validate(&sample, 2)?;
  reject_constant(&sample, "zero spread")?;
  let (mut sorted, unit) = unit_scaled(sample.into_owned());
  let nf = sorted.len() as f64;
  let mean = sorted.iter().sum::<f64>() / nf;
  let std = (sorted.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (nf - 1.0)).sqrt();
  sorted.sort_by(f64::nan_last);
  let iqr = quantile_sorted(&sorted, 0.75) - quantile_sorted(&sorted, 0.25);
  let scale = if iqr > 0.0 { std.min(iqr / 1.34) } else { std };
  let h = 0.9 * scale * nf.powf(-1.0 / 5.0) / unit;
  if h.is_finite() && h > 0.0 {
    Ok(h)
  } else {
    Err(StatsError::Numerical {
      stage: "silverman bandwidth",
    })
  }
}

/// Linearly interpolated sample quantile at `p ∈ [0, 1]`; NaN outside it, on an empty sample or on
/// a NaN.
pub fn quantile(data: ArrayView1<f64>, p: f64) -> f64 {
  if data.is_empty() || !(0.0..=1.0).contains(&p) || data.iter().any(|v| v.is_nan()) {
    return f64::NAN;
  }
  let mut sorted = data.to_vec();
  sorted.sort_by(f64::nan_last);
  quantile_sorted(&sorted, p)
}

/// An exact rank or equal neighbours return the element, so an infinity is not interpolated into
/// NaN.
fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
  let rank = p * (sorted.len() as f64 - 1.0);
  let lower = rank.floor() as usize;
  let upper = rank.ceil() as usize;
  if lower == upper || sorted[lower] == sorted[upper] {
    return sorted[lower];
  }
  let weight = rank - lower as f64;
  sorted[lower] + weight * (sorted[upper] - sorted[lower])
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use ndarray::array;

  use super::*;
  use crate::StatsError;

  #[test]
  fn quantile_interpolates_and_answers_nan_outside_the_unit_interval() {
    let data = array![3.0, 1.0, 2.0, 4.0];
    assert_eq!(quantile(data.view(), 0.0), 1.0);
    assert_eq!(quantile(data.view(), 1.0), 4.0);
    assert_eq!(quantile(data.view(), 0.5), 2.5);
    assert!(quantile(data.view(), 1.5).is_nan());
    assert!(quantile(Array1::<f64>::zeros(0).view(), 0.5).is_nan());
    assert!(quantile(array![1.0, f64::NAN].view(), 0.5).is_nan());
    assert_eq!(
      quantile(array![1.0, f64::INFINITY].view(), 1.0),
      f64::INFINITY
    );
    let tail = array![1.0, f64::INFINITY, f64::INFINITY];
    assert_eq!(quantile(tail.view(), 0.75), f64::INFINITY);
  }

  #[test]
  fn silverman_needs_two_spread_points() {
    assert_eq!(
      silverman_bandwidth(array![1.0].view()).unwrap_err(),
      StatsError::InsufficientData { needed: 2, got: 1 }
    );
    assert_eq!(
      silverman_bandwidth(array![2.0, 2.0, 2.0].view()).unwrap_err(),
      StatsError::DegenerateData("zero spread")
    );
    let h = silverman_bandwidth(array![1.0, 2.0, 3.0, 4.0, 10.0].view()).unwrap();
    assert!(h > 0.0 && h.is_finite());
  }

  #[test]
  fn silverman_reads_the_standard_deviation_alone_when_the_interquartile_range_is_zero() {
    let data = array![1.0_f64, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0];
    let mean = data.sum() / 8.0;
    let std = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / 7.0).sqrt();
    assert_eq!(
      silverman_bandwidth(data.view()).unwrap(),
      0.9 * std * 8.0_f64.powf(-0.2)
    );
  }

  #[test]
  #[should_panic(expected = "bandwidth must satisfy `bandwidth > 0.0`")]
  fn a_non_positive_bandwidth_is_a_precondition() {
    let _ = GaussianKde::new(array![1.0, 2.0].view(), 0.0);
  }

  #[test]
  fn an_empty_or_non_finite_sample_is_an_error() {
    assert_eq!(
      GaussianKde::new(Array1::<f64>::zeros(0).view(), 0.5).unwrap_err(),
      StatsError::InsufficientData { needed: 1, got: 0 }
    );
    assert!(matches!(
      GaussianKde::new(array![1.0, f64::NAN].view(), 0.5),
      Err(StatsError::InvalidObservation { index: 1, .. })
    ));
    assert!(matches!(
      GaussianKde::with_silverman_bandwidth(array![1.0, 2.0, f64::INFINITY].view()),
      Err(StatsError::InvalidObservation { index: 2, .. })
    ));
  }

  #[test]
  fn the_density_integrates_to_one_on_a_wide_grid() {
    let kde = GaussianKde::new(array![1.0, 2.0, 3.0].view(), 0.5).unwrap();
    let xs = Array1::linspace(-10.0, 14.0, 24_001);
    let ys = kde.evaluate_array(xs.view());
    let mass = ys.sum() * (24.0 / 24_000.0);
    assert!((mass - 1.0).abs() < 1e-6, "{mass}");
    assert_eq!(kde.bandwidth(), 0.5);
  }
}
