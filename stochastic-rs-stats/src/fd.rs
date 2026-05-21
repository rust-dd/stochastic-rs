//! Deprecated v2.2 fractal-dimension / Hurst convenience API.
//!
//! Use [`crate::fractal_dim`] for pure fractal-dim estimation and
//! [`crate::hurst::from_prices`] for Hurst-from-prices convenience.

#![allow(deprecated)]

use ndarray::ArrayView1;

pub use crate::fractal_dim::FdError;

/// Deprecated alias for [`crate::fractal_dim::FractalDim<f64>`].
#[deprecated(
  since = "2.3.0",
  note = "use `stochastic_rs_stats::fractal_dim::FractalDim`"
)]
pub type FractalDim = crate::fractal_dim::FractalDim<f64>;

/// Deprecated: use [`crate::hurst::from_prices::estimate_hurst`].
#[deprecated(
  since = "2.3.0",
  note = "use `stochastic_rs_stats::hurst::from_prices::estimate_hurst`"
)]
pub fn estimate_hurst(closes: ArrayView1<f64>) -> f64 {
  crate::hurst::from_prices::estimate_hurst::<f64>(closes)
}

/// Deprecated: use [`crate::hurst::from_prices::hurst_from_signal`].
#[deprecated(
  since = "2.3.0",
  note = "use `stochastic_rs_stats::hurst::from_prices::hurst_from_signal`"
)]
pub fn hurst_from_signal(signal: ArrayView1<f64>) -> f64 {
  crate::hurst::from_prices::hurst_from_signal::<f64>(signal)
}
