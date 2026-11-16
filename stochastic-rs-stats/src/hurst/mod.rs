//! # Hurst exponent estimators
//!
//! Unified [`HurstEstimator`] trait plus per-method config structs.  Each
//! estimator returns a [`HurstFit`] carrying the point estimate, an
//! optional standard error, sample size, and a method-specific
//! [`HurstDiagnostic`] (log-log slope, variation ratio, …).
//!
//! Available estimators (in this crate):
//!
//! | Struct                                  | Method                                              |
//! |-----------------------------------------|-----------------------------------------------------|
//! | [`rs::RescaledRange`]                   | Hurst (1951) + Anis-Lloyd (1976) bias correction     |
//! | [`dfa::Dfa`]                            | Peng et al. (1994) detrended fluctuation analysis    |
//! | [`gph::Gph`]                            | Geweke-Porter-Hudak (1983) log-periodogram           |
//! | [`wavelet::Wavelet`]                    | Veitch-Abry (1999) DWT + WLS regression              |
//! | [`variations::Variations`]              | Daubechies / central-diff / `k`-th p-power variation |
//! | [`crate::fractal_dim::Higuchi`]         | Higuchi (1988) FD adapter, `H = 2 - D`               |
//! | [`crate::fractal_dim::Variogram`]       | Variogram FD adapter (Constantine-Hall 1994)         |
//!
//! The fractal-dimension estimators ([`crate::fractal_dim::Higuchi`],
//! [`crate::fractal_dim::Variogram`]) implement **both**
//! [`crate::fractal_dim::FractalDimEstimator`] (returns `D`) and
//! [`HurstEstimator`] (returns `H = 2 - D`, unclamped).  Pick
//! which output you want by importing the relevant trait or by
//! disambiguating the call:
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
//!
//! ## Choosing an estimator
//!
//! All seven estimate the same thing — the self-similarity exponent of
//! the series handed to `.estimate()` — by different routes;
//! [`Whittle::fit`] estimates something else entirely. Read the estimand
//! before the estimator name.
//!
//! **Path self-similarity** ([`RescaledRange`], [`Dfa`], [`Gph`],
//! [`Wavelet`], [`Variations`]): all five fit a log-log (or
//! log-frequency) regression directly on the series, after optional
//! first-differencing. They are not interchangeable up to tuning — the
//! literature documents real differences:
//!
//! - [`RescaledRange`] is the oldest (Hurst 1951) and, per Weron (2002) —
//!   already the citation backing this struct's Anis-Lloyd correction —
//!   the least reliable of the group in finite samples. Kept for its
//!   historical role and cheap computation, not as the recommended
//!   default.
//! - [`Dfa`] detrends every window with a degree-`order` polynomial before
//!   measuring fluctuation, which is specifically why Peng et al. (1994)
//!   and Kantelhardt et al. (2001) built it: robustness to the smooth
//!   trends that make [`RescaledRange`] unreliable on real (as opposed to
//!   simulated stationary) data.
//! - [`Gph`] and [`Wavelet`] work in the frequency/scale domain and are
//!   the only two that populate `std_err` with a closed-form asymptotic
//!   value rather than leaving it `None` — reach for one of these two
//!   when a confidence interval matters, not just a point estimate.
//!   [`Wavelet`]'s `Daubechies4` filter has 2 vanishing moments, so it is
//!   blind to a linear trend but not a quadratic one; [`Dfa`] with a
//!   higher `order` handles trends [`Wavelet`] cannot.
//! - [`Variations`] (Coeurjolly 2001; Brouste & Iacus 2013, arXiv:1112.3777)
//!   is the one derived for a discretely-observed **fractional
//!   Ornstein-Uhlenbeck**-type process rather than pure fBM/fGN
//!   self-similarity — reach for it, specifically
//!   `VariationKind::PowerVariation`, when the generating model is
//!   believed to be mean-reverting-rough rather than a pure fractional
//!   random walk. Its three [`VariationKind`] values *are* interchangeable
//!   in the differ-only-in-tuning sense: the same log-ratio-of-two-scales
//!   idea, different filter/kernel.
//!
//! **Fractal-dimension route** ([`crate::fractal_dim::Higuchi`],
//! [`crate::fractal_dim::Variogram`]): these reach `H` from a different
//! literature entirely — curve-length (Higuchi 1988, biophysical signal
//! processing) and variogram-ratio (Constantine & Hall 1994, spatial
//! statistics) fractal-dimension estimation, converted via the FBM
//! identity `H = 2 - D`. [`fd_adapters`] reports `H = 2 - D` unclamped, so a noisy fit
//! that pushes `D` outside `[1, 2]` shows as an `H` outside `[0, 1]`.
//!
//! **A different quantity entirely** ([`Whittle::fit`]): Fukasawa,
//! Takabatake & Westphal (2019, arXiv:1905.04852) estimate the roughness of
//! a **latent volatility** process from log realized variance aggregated
//! from intraday returns — not the self-similarity exponent of a series
//! sampled directly from, say, [`stochastic_rs_stochastic::noise::fgn::Fgn`]
//! — so [`Whittle`] is not a [`HurstEstimator`] and returns a
//! [`FukasawaFit`].
//!
//! **Data requirements** — hard validation floor, all generic over
//! `T: `[`crate::traits::FloatExt`] (`f32` or `f64`):
//!
//! | Estimator | Min. obs. (defaults) |
//! |---|---|
//! | [`RescaledRange`] | `4 (min_window + 1)` → 68 (`2 max(min_window + 1, 8)` with `max_window`) |
//! | [`Dfa`] | `4 (min_window + 1)` → 36 (`max(4 min_window, order + 4)` with `max_window`) |
//! | [`Gph`] | 64 |
//! | [`Wavelet`] | `max(64, 2^(j_min + 3) + 1)` → 64 |
//! | [`Variations`] | 5 (`CentralDiff`, the default) to 8 (`Daubechies`), or `2k + 1` for `PowerVariation { k, .. }` |
//! | [`crate::fractal_dim::Higuchi`] | 3 (but see the `kmax` caveat above — the hard floor is far below what a meaningful log-log regression needs) |
//! | [`crate::fractal_dim::Variogram`] | 3 (same caveat) |

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

/// Method-specific auxiliary information returned alongside the Hurst
/// estimate.
///
/// Carrying these as an enum keeps the [`HurstFit`] struct flat for
/// the common `.hurst` access while still letting callers pattern-match
/// on the variant to recover slope, residuals, or other diagnostics.
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
  /// Fractal-dimension derived Hurst, carrying the underlying `D`
  /// value (used by [`crate::fractal_dim::Higuchi`] /
  /// [`crate::fractal_dim::Variogram`] when invoked through the
  /// [`HurstEstimator`] trait).
  FractalDim { d: T },
  /// No diagnostic available.
  None,
}

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
