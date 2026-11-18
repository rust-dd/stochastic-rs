//! Fractal-dimension estimators behind `FractalDimEstimator`: Higuchi curve length and variogram.
//!
//! [`Higuchi`] and [`Variogram`] also answer `H = 2 − D` through [`crate::hurst::HurstEstimator`].

use ndarray::ArrayView1;

use crate::StatsError;
use crate::traits::FloatExt;

pub mod higuchi;
pub mod variogram;

pub use higuchi::Higuchi;
pub use variogram::Variogram;

/// A fractal-dimension estimator: it owns its configuration and reads one series into an [`FdFit`];
/// implemented by [`Higuchi`] and [`Variogram`].
pub trait FractalDimEstimator<T: FloatExt> {
  fn estimate(&self, x: ArrayView1<T>) -> Result<FdFit<T>, StatsError>;
}

/// Point estimate plus method-specific diagnostics for the fractal
/// dimension `D`.
#[derive(Clone, Debug)]
pub struct FdFit<T: FloatExt = f64> {
  /// Point estimate of the fractal dimension.
  pub d: T,
  /// Number of observations consumed.
  pub nobs: usize,
  /// Method-specific auxiliary information.
  pub diagnostic: FdDiagnostic<T>,
}

/// Method-specific auxiliary information returned alongside the
/// fractal-dimension estimate.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum FdDiagnostic<T: FloatExt> {
  /// Higuchi log-log regression diagnostics.
  LogLogRegression {
    slope: T,
    intercept: T,
    r_squared: T,
    log_scales: Vec<T>,
    log_stats: Vec<T>,
  },
  /// Variogram-ratio diagnostics.
  VariogramRatio { v_short: T, v_long: T },
  /// No diagnostic available.
  None,
}
