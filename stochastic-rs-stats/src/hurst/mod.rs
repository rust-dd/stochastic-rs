//! Hurst-exponent estimators behind one `HurstEstimator` trait; choosing one is on the stats page.
//!
//! The fractal-dimension estimators implement it too, as `H = 2 − D`: name the trait in the call.
//!
//! ```
//! use stochastic_rs_core::simd_rng::Deterministic;
//! use stochastic_rs_stats::hurst::HurstEstimator;
//! use stochastic_rs_stochastic::process::fbm::Fbm;
//! use stochastic_rs_stochastic::traits::ProcessExt;
//!
//! let signal = Fbm::new(0.7_f64, 256, Some(1.0), Deterministic::new(7)).sample();
//! let h = stochastic_rs_stats::fractal_dim::Higuchi { kmax: 32 }
//!     .estimate(signal.view())  // HurstEstimator::estimate → HurstFit
//!     .unwrap()
//!     .hurst;
//! assert!((0.0..=1.0).contains(&h));
//! ```

use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_stochastic::device::Cpu;
use stochastic_rs_stochastic::noise::fgn::Fgn;
use stochastic_rs_stochastic::process::fbm::Fbm;

use crate::StatsError;
use crate::traits::FloatExt;

pub mod dfa;
pub mod fd_adapters;
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
pub use whittle::FukasawaFit;
pub use whittle::Whittle;

/// The self-similarity exponent of the series itself: [`RescaledRange`], [`Dfa`], [`Gph`],
/// [`Wavelet`], [`Variations`], and `H = 2 − D` from [`crate::fractal_dim`]'s two estimators.
pub trait HurstEstimator<T: FloatExt> {
  /// `H` of `x`; `Err` on a constant series, a configuration out of range or too few points.
  fn estimate(&self, x: ArrayView1<T>) -> Result<HurstFit<T>, StatsError>;
}

/// Point estimate plus method-specific diagnostics for the Hurst
/// exponent.
#[derive(Clone, Debug)]
pub struct HurstFit<T: FloatExt = f64> {
  /// Point estimate of the Hurst exponent.
  pub hurst: T,
  /// Asymptotic standard error when available (currently only [`Gph`]
  /// and [`Wavelet`] populate this; others leave it `None`).
  pub std_err: Option<T>,
  /// Number of observations consumed.
  pub nobs: usize,
  /// Method-specific auxiliary information (log-log regression
  /// coefficients, variation ratio, …).
  pub diagnostic: HurstDiagnostic<T>,
}

impl<T: FloatExt> HurstFit<T> {
  /// Fractional Gaussian noise at this `hurst`; chain `.with_backend(handle)` for a device. Panics
  /// where `Fgn::new` does, so on a `hurst` outside `[0, 1]`, which an unclamped `2 − D` can give.
  pub fn to_fgn<S: SeedExt>(&self, n: usize, t: Option<T>, seed: S) -> Fgn<T, S, Cpu> {
    Fgn::new(self.hurst, n, t, seed)
  }

  /// Fractional Brownian motion at this `hurst`; chain `.with_backend(handle)` for a device. Panics
  /// where `Fbm::new` does, so on a `hurst` outside `[0, 1]`, which an unclamped `2 − D` can give.
  pub fn to_fbm<S: SeedExt>(&self, n: usize, t: Option<T>, seed: S) -> Fbm<T, S, Cpu> {
    Fbm::new(self.hurst, n, t, seed)
  }
}

/// Method-specific diagnostics beside the Hurst estimate: an enum, so that [`HurstFit`] stays flat
/// and a caller matches on the variant for the slope or the other values.
#[derive(Clone, Debug)]
#[non_exhaustive]
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
  /// k-th order p-power variation ratio diagnostics.
  Variations {
    /// Variation at the short stride.
    v_short: T,
    /// Variation at the long stride.
    v_long: T,
  },
  /// The fractal dimension `D` behind `H = 2 − D`, from [`crate::fractal_dim::Higuchi`] or
  /// [`crate::fractal_dim::Variogram`] through [`HurstEstimator`].
  FractalDim { d: T },
  /// No diagnostic available.
  None,
}

/// Weighted least squares of `y` on `x` (OLS when `weights` is `None`): `(slope, intercept, r²)`,
/// or `None` for a degenerate fit.
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
