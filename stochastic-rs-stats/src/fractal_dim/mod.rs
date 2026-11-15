//! # Fractal dimension estimators
//!
//! Unified [`FractalDimEstimator`] trait plus per-method config struct,
//! mirroring [`crate::hurst::HurstEstimator`].  Each estimator returns a
//! [`FdResult`] carrying the point estimate `D`, sample size, and a
//! method-specific [`FdDiagnostic`] (log-log slope diagnostics for
//! Higuchi, variogram-ratio diagnostics for variogram).
//!
//! Available estimators:
//!
//! | Struct                        | Method                                                 |
//! |-------------------------------|--------------------------------------------------------|
//! | [`higuchi::Higuchi`]          | Higuchi (1988) curve-length method                     |
//! | [`variogram::Variogram`]      | Variogram-ratio (Constantine-Hall 1994), `p`-power     |
//!
//! Both also implement [`crate::hurst::HurstEstimator`] via
//! `crate::hurst::fd_adapters`, so callers who want `H = 2 - D`
//! directly can disambiguate the call by importing the relevant trait.

use ndarray::ArrayView1;

use crate::StatsError;
use crate::traits::FloatExt;

pub mod higuchi;
pub mod variogram;

pub use higuchi::Higuchi;
pub use variogram::Variogram;

/// Unified fractal-dimension estimator interface.
///
/// Each estimator owns its configuration (kmax, p-norm, …) and exposes
/// a single [`estimate`] entry point that consumes a contiguous time
/// series and produces a [`FdResult`].
///
/// Concrete impls in this crate: [`Higuchi`], [`Variogram`].
///
/// [`estimate`]: FractalDimEstimator::estimate
pub trait FractalDimEstimator<T: FloatExt> {
  fn estimate(&self, x: ArrayView1<T>) -> Result<FdResult<T>, StatsError>;
}

/// Point estimate plus method-specific diagnostics for the fractal
/// dimension `D`.
#[derive(Clone, Debug)]
pub struct FdResult<T: FloatExt = f64> {
  /// Point estimate of the fractal dimension.
  pub d: T,
  /// Number of observations consumed.
  pub n_obs: usize,
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

/// Convenience: compute Higuchi FD directly from a view.  Free-function
/// shortcut that bypasses the trait + result struct.
pub fn higuchi<T: FloatExt>(x: ArrayView1<T>, kmax: usize) -> Result<T, StatsError> {
  higuchi::compute(x, kmax)
}

/// Convenience: compute variogram FD directly from a view.  Free-function
/// shortcut that bypasses the trait + result struct.
pub fn variogram<T: FloatExt>(x: ArrayView1<T>, p: T) -> Result<T, StatsError> {
  variogram::compute(x, p)
}
