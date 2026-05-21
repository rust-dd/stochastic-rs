//! # Hurst exponent estimators
//!
//! Unified [`HurstEstimator`] trait plus per-method config structs.  Each
//! estimator returns a [`HurstResult`] carrying the point estimate, an
//! optional standard error, sample size, and a method-specific
//! [`HurstDiagnostic`] (log-log slope, Whittle objective value, …).
//!
//! Available estimators (in this crate):
//!
//! | Struct                                  | Method                                              |
//! |-----------------------------------------|-----------------------------------------------------|
//! | [`rs::RescaledRange`]                   | Hurst (1951) + Anis-Lloyd (1976) bias correction     |
//! | [`dfa::Dfa`]                            | Peng et al. (1994) detrended fluctuation analysis    |
//! | [`gph::Gph`]                            | Geweke-Porter-Hudak (1983) log-periodogram           |
//! | [`wavelet::Wavelet`]                    | Veitch-Abry (1999) DWT + WLS regression              |
//! | [`whittle::Whittle`]                    | Fukasawa et al. (2019) adapted Whittle for rough vol |
//! | [`variations::Variations`]              | Daubechies / central-diff / `k`-th p-power variation |
//! | [`crate::fractal_dim::Higuchi`]         | Higuchi (1988) FD adapter, `H = 2 - D`               |
//! | [`crate::fractal_dim::Variogram`]       | Variogram FD adapter (Constantine-Hall 1994)         |
//!
//! The fractal-dimension estimators ([`crate::fractal_dim::Higuchi`],
//! [`crate::fractal_dim::Variogram`]) implement **both**
//! [`crate::fractal_dim::FractalDimEstimator`] (returns `D`) and
//! [`HurstEstimator`] (returns `H = 2 - D` clamped to `(0, 1)`).  Pick
//! which output you want by importing the relevant trait or by
//! disambiguating the call:
//!
//! ```ignore
//! use stochastic_rs_stats::hurst::HurstEstimator;
//! let h = stochastic_rs_stats::fractal_dim::Higuchi { kmax: 32 }
//!     .estimate(signal)?  // HurstEstimator::estimate → HurstResult
//!     .hurst;
//! ```

use std::fmt;

use ndarray::ArrayView1;

use crate::traits::FloatExt;

pub mod dfa;
pub mod fd_adapters;
pub mod from_prices;
pub mod gph;
pub mod rs;
pub mod variations;
pub mod wavelet;
pub mod whittle;

pub use dfa::Dfa;
pub use gph::Gph;
pub use rs::RescaledRange;
pub use variations::VariationKind;
pub use variations::Variations;
pub use wavelet::Wavelet;
pub use wavelet::WaveletKind;
pub use whittle::FukasawaResult;
pub use whittle::Whittle;

/// Unified Hurst-estimator interface.
///
/// Each estimator owns its configuration (window bounds, polynomial
/// order, bandwidth, …) and exposes a single [`estimate`] entry point
/// that consumes a contiguous time series and produces a
/// [`HurstResult`].
///
/// Concrete impls in this crate: [`RescaledRange`], [`Dfa`], [`Gph`],
/// [`Wavelet`], [`Whittle`], [`Variations`].
///
/// [`estimate`]: HurstEstimator::estimate
pub trait HurstEstimator<T: FloatExt> {
  fn estimate(&self, x: ArrayView1<T>) -> Result<HurstResult<T>, HurstError>;
}

/// Point estimate plus method-specific diagnostics for the Hurst
/// exponent.
#[derive(Clone, Debug)]
pub struct HurstResult<T: FloatExt = f64> {
  /// Point estimate of the Hurst exponent.
  pub hurst: T,
  /// Asymptotic standard error when available (currently only [`Gph`]
  /// and [`Wavelet`] populate this; others leave it `None`).
  pub std_err: Option<T>,
  /// Number of observations consumed.
  pub n_obs: usize,
  /// Method-specific auxiliary information (log-log regression
  /// coefficients, Whittle objective, …).
  pub diagnostic: HurstDiagnostic<T>,
}

/// Method-specific auxiliary information returned alongside the Hurst
/// estimate.
///
/// Carrying these as an enum keeps the [`HurstResult`] struct flat for
/// the common `.hurst` access while still letting callers pattern-match
/// on the variant to recover slope, residuals, or other diagnostics.
#[derive(Clone, Debug)]
pub enum HurstDiagnostic<T: FloatExt> {
  /// Log-log regression diagnostics (R/S, DFA, GPH, Wavelet).
  LogLogRegression {
    /// Slope of `log(stat) ~ slope · log(scale)`.
    slope: T,
    /// Regression intercept.
    intercept: T,
    /// Coefficient of determination.
    r_squared: T,
    /// Log-scales used in the regression.
    log_scales: Vec<T>,
    /// Log-statistics matched 1:1 with `log_scales`.
    log_stats: Vec<T>,
  },
  /// Whittle quasi-likelihood diagnostics (Fukasawa).
  Whittle {
    /// Final negative-log-likelihood value.
    neg_log_lik: T,
    /// Recovered `η = ν · δ⁻ᴴ` scale parameter.
    eta: T,
  },
  /// k-th order p-power variation ratio diagnostics.
  Variations {
    /// Variation at the short stride.
    v_short: T,
    /// Variation at the long stride.
    v_long: T,
  },
  /// Fractal-dimension derived Hurst, carrying the underlying `D`
  /// value (used by [`crate::fractal_dim::Higuchi`] /
  /// [`crate::fractal_dim::Variogram`] when invoked through the
  /// [`HurstEstimator`] trait).
  FractalDim { d: T },
  /// No diagnostic available.
  None,
}

/// Errors returned by [`HurstEstimator::estimate`].
#[derive(Debug, Clone, PartialEq)]
pub enum HurstError {
  /// Input series is too short for the chosen estimator.
  TooFewObservations { got: usize, required: usize },
  /// Input series is constant or has zero variance.
  DegeneratePath,
  /// A configuration parameter is outside its admissible range.
  InvalidParameter(&'static str, f64),
  /// Log-log regression has fewer than two valid scale points.
  NotEnoughScales,
  /// Underlying linear regression failed.
  RegressionFailed,
  /// Numerical optimisation failed (Whittle and similar).
  OptimizationFailed,
}

impl fmt::Display for HurstError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      HurstError::TooFewObservations { got, required } => {
        write!(
          f,
          "input has {} points, estimator needs at least {}",
          got, required
        )
      }
      HurstError::DegeneratePath => write!(f, "path is constant or has zero variance"),
      HurstError::InvalidParameter(name, value) => {
        write!(f, "invalid parameter `{}`: {}", name, value)
      }
      HurstError::NotEnoughScales => {
        write!(f, "fewer than two valid scales for log-log regression")
      }
      HurstError::RegressionFailed => write!(f, "log-log regression failed"),
      HurstError::OptimizationFailed => write!(f, "Hurst optimisation failed to converge"),
    }
  }
}

impl std::error::Error for HurstError {}

/// Weighted-least-squares regression `y ~ slope * x + intercept`.
///
/// Pass `weights = None` to fall back to OLS.  Returns
/// `(slope, intercept, r_squared)` or `None` for a degenerate fit.
pub(crate) fn weighted_linreg(
  x: &[f64],
  y: &[f64],
  weights: Option<&[f64]>,
) -> Option<(f64, f64, f64)> {
  assert_eq!(x.len(), y.len());
  if let Some(w) = weights {
    assert_eq!(w.len(), x.len());
  }
  if x.len() < 2 {
    return None;
  }

  let n = x.len();
  let w_sum = match weights {
    Some(w) => w.iter().sum::<f64>(),
    None => n as f64,
  };
  if !w_sum.is_finite() || w_sum <= 0.0 {
    return None;
  }

  let mean = |v: &[f64]| -> f64 {
    let acc = match weights {
      Some(w) => v.iter().zip(w.iter()).map(|(a, w)| *a * *w).sum::<f64>(),
      None => v.iter().sum::<f64>(),
    };
    acc / w_sum
  };

  let xm = mean(x);
  let ym = mean(y);

  let mut sxy = 0.0;
  let mut sxx = 0.0;
  let mut syy = 0.0;
  for i in 0..n {
    let w = weights.map(|w| w[i]).unwrap_or(1.0);
    let dx = x[i] - xm;
    let dy = y[i] - ym;
    sxy += w * dx * dy;
    sxx += w * dx * dx;
    syy += w * dy * dy;
  }
  if sxx <= 0.0 || !sxx.is_finite() {
    return None;
  }
  let slope = sxy / sxx;
  let intercept = ym - slope * xm;
  let r_squared = if syy > 0.0 {
    (sxy * sxy) / (sxx * syy)
  } else {
    0.0
  };
  Some((slope, intercept, r_squared))
}

/// Generate `n` log-spaced integer window sizes between `min` and `max`
/// inclusive, deduplicated and sorted ascending.
pub(crate) fn log_spaced_windows(min: usize, max: usize, n: usize) -> Vec<usize> {
  if max <= min || n < 2 {
    return vec![min, max].into_iter().filter(|s| *s >= 2).collect();
  }
  let lo = (min as f64).ln();
  let hi = (max as f64).ln();
  let mut out = Vec::with_capacity(n);
  for i in 0..n {
    let t = i as f64 / (n - 1) as f64;
    let s = (lo + t * (hi - lo)).exp().round() as usize;
    if s >= 2 {
      out.push(s);
    }
  }
  out.sort_unstable();
  out.dedup();
  out
}

/// Read `x` as a contiguous `f64` slice when possible, else copy.
pub(crate) fn to_f64_vec<T: FloatExt>(x: ArrayView1<T>) -> Vec<f64> {
  x.iter().map(|v| v.to_f64().unwrap_or(f64::NAN)).collect()
}
