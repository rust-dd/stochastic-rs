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
//! For Hurst-exponent estimation use [`crate::hurst`] estimators
//! directly — the `D = 2 - H` conversion adapter is no longer wired
//! into the Hurst trait, since the natural output of Higuchi/Variogram
//! is `D`, not `H`.  Convert manually when needed.

use std::fmt;

use ndarray::Array1;
use ndarray::ArrayView1;

use crate::traits::FloatExt;

pub mod higuchi;
pub mod variogram;

pub use higuchi::Higuchi;
pub use variogram::Variogram;

/// Errors returned by fractal-dimension estimators.
#[derive(Debug, Clone, PartialEq)]
pub enum FdError {
  /// Path has fewer points than the estimator needs.
  PathTooShort { got: usize, required: usize },
  /// `p` parameter must be strictly positive.
  NonPositiveP(f64),
  /// `kmax` parameter must be at least 2 for Higuchi.
  KmaxTooSmall(usize),
  /// Variogram or curve-length computation produced a non-finite or
  /// non-positive value (e.g. constant path, all-zero increments).
  DegeneratePath,
  /// Higuchi regression has fewer than two valid scales after filtering.
  NotEnoughScales,
  /// Underlying linear regression failed.
  RegressionFailed,
}

impl fmt::Display for FdError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      FdError::PathTooShort { got, required } => write!(
        f,
        "path has {} points, fractal-dim estimator needs at least {}",
        got, required
      ),
      FdError::NonPositiveP(p) => write!(f, "p must be positive (got {})", p),
      FdError::KmaxTooSmall(k) => {
        write!(f, "kmax must be at least 2 for Higuchi (got {})", k)
      }
      FdError::DegeneratePath => write!(f, "variogram is undefined for degenerate path"),
      FdError::NotEnoughScales => write!(f, "Higuchi regression has fewer than 2 valid scales"),
      FdError::RegressionFailed => write!(f, "linear regression failed"),
    }
  }
}

impl std::error::Error for FdError {}

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
  fn estimate(&self, x: ArrayView1<T>) -> Result<FdResult<T>, FdError>;
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

/// **Deprecated** convenience wrapper, retained for v2.2 API
/// compatibility.  New code should use [`Higuchi`] /
/// [`Variogram`] directly through the [`FractalDimEstimator`] trait.
#[derive(Clone, Debug)]
pub struct FractalDim<T: FloatExt> {
  pub x: Array1<T>,
}

impl<T: FloatExt> FractalDim<T> {
  #[must_use]
  pub fn new(x: Array1<T>) -> Self {
    Self { x }
  }

  /// Variogram-based fractal dimension (p-power).  `p = None` defaults
  /// to `1.0`.
  pub fn variogram(&self, p: Option<T>) -> Result<T, FdError> {
    let p = p.unwrap_or_else(|| T::from_f64_fast(1.0));
    variogram::compute(self.x.view(), p)
  }

  /// Higuchi (1988) curve-length fractal dimension.
  pub fn higuchi_fd(&self, kmax: usize) -> Result<T, FdError> {
    higuchi::compute(self.x.view(), kmax)
  }
}

/// Convenience: compute Higuchi FD directly from a view.  Free-function
/// shortcut that bypasses the trait + result struct.
pub fn higuchi<T: FloatExt>(x: ArrayView1<T>, kmax: usize) -> Result<T, FdError> {
  higuchi::compute(x, kmax)
}

/// Convenience: compute variogram FD directly from a view.  Free-function
/// shortcut that bypasses the trait + result struct.
pub fn variogram<T: FloatExt>(x: ArrayView1<T>, p: T) -> Result<T, FdError> {
  variogram::compute(x, p)
}
