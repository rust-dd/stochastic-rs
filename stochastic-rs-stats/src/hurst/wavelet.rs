//! Veitch-Abry (1999) wavelet-based Hurst estimator.
//!
//! 1. Apply the pyramidal discrete wavelet transform with a
//!    compactly-supported orthonormal wavelet (currently Daubechies-4)
//!    using periodic boundary extension.
//! 2. At each dyadic scale `j` compute the sample energy
//!    `μⱼ = (1/Nⱼ) Σₖ d²_{j,k}` of the detail coefficients.
//! 3. Subtract the Eq. 21 bias correction
//!    `gⱼ = (ψ(Nⱼ/2) - ln(Nⱼ/2)) / ln 2` so the residual is unbiased
//!    in expectation under Gaussianity.
//! 4. Weighted least-squares regression `log₂ μⱼ - gⱼ ~ slope · j + c`
//!    with weights `wⱼ = Nⱼ (ln 2)² / 2`.
//! 5. Slope inversion: FBM input → `H = (slope - 1) / 2`; FGN input →
//!    `H = (slope + 1) / 2`.
//!
//! Standard error: `SE(H) = √(2 / (slope-variance · Σ wⱼ))`
//! (asymptotic Veitch-Abry result, simplified to the unweighted
//! variance form).
//!
//! References:
//! - Veitch D. & Abry P. (1999) — *A wavelet-based joint estimator of
//!   the parameters of long-range dependence*, IEEE T-IT 45(3),
//!   878–897, DOI: 10.1109/18.761330.
//! - Abry P., Flandrin P., Taqqu M. S., Veitch D. (2003) — *Self-
//!   similarity and long-range dependence through the wavelet lens*.

use std::f64::consts::LN_2;

use ndarray::ArrayView1;
use stochastic_rs_distributions::special::digamma;

use super::HurstDiagnostic;
use super::HurstError;
use super::HurstEstimator;
use super::HurstResult;
use super::to_f64_vec;
use super::weighted_linreg;
use crate::traits::FloatExt;

/// Standard Daubechies-4 scaling (low-pass) filter coefficients,
/// normalised so that `Σ hₖ = √2` and `Σ h²ₖ = 1`.  Distinct from
/// [`super::variations::daubechies_coeffs`] (Coeurjolly variant,
/// different normalisation and sign pattern).
fn daubechies4_scaling() -> Vec<f64> {
  vec![
    0.482_962_913_144_534_1,
    0.836_516_303_737_807_9,
    0.224_143_868_042_013_4,
    -0.129_409_522_551_260_4,
  ]
}

/// Wavelet family used by [`Wavelet`].  Currently only Daubechies-4.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WaveletKind {
  Daubechies4,
}

/// Veitch-Abry wavelet Hurst estimator.
///
/// The estimator is defined on stationary FGN-like input.  When
/// `take_differences = true` (default) the input is first-differenced
/// internally so FBM-like walks can be passed directly.  Slope-to-`H`
/// map is always the FGN form `H = (slope + 1) / 2`.
#[derive(Clone, Debug)]
pub struct Wavelet {
  pub wavelet: WaveletKind,
  /// First-difference the input before estimation (set `false` when
  /// the input is already a stationary FGN-like series).
  pub take_differences: bool,
  /// Smallest dyadic scale included in the regression (1-based).
  pub j_min: usize,
  /// Largest dyadic scale (`None` → `⌊log₂ N⌋ - 2`).
  pub j_max: Option<usize>,
}

impl Default for Wavelet {
  fn default() -> Self {
    Self {
      wavelet: WaveletKind::Daubechies4,
      take_differences: true,
      j_min: 2,
      j_max: None,
    }
  }
}

impl Wavelet {
  #[must_use]
  pub fn new() -> Self {
    Self::default()
  }
}

impl<T: FloatExt> HurstEstimator<T> for Wavelet {
  fn estimate(&self, x: ArrayView1<T>) -> Result<HurstResult<T>, HurstError> {
    let n_obs = x.len();
    if n_obs < 64 {
      return Err(HurstError::TooFewObservations {
        got: n_obs,
        required: 64,
      });
    }
    if self.j_min < 1 {
      return Err(HurstError::InvalidParameter("j_min", self.j_min as f64));
    }

    let signal_raw = to_f64_vec::<T>(x);
    let signal: Vec<f64> = if self.take_differences {
      signal_raw.windows(2).map(|w| w[1] - w[0]).collect()
    } else {
      signal_raw
    };
    let n_series = signal.len();
    if n_series < 32 {
      return Err(HurstError::TooFewObservations {
        got: n_series,
        required: 32,
      });
    }
    let (h_filter, g_filter) = match self.wavelet {
      WaveletKind::Daubechies4 => qmf_from_low(&daubechies4_scaling()),
    };

    let max_levels = (n_series as f64).log2().floor() as usize;
    let j_cap = self
      .j_max
      .unwrap_or(max_levels.saturating_sub(2))
      .min(max_levels);
    if j_cap < self.j_min + 1 {
      return Err(HurstError::NotEnoughScales);
    }

    let levels = pyramidal_dwt(&signal, &h_filter, &g_filter, j_cap);
    if levels.len() < 2 {
      return Err(HurstError::NotEnoughScales);
    }

    let mut log_j = Vec::with_capacity(levels.len());
    let mut log_mu_corr = Vec::with_capacity(levels.len());
    let mut weights = Vec::with_capacity(levels.len());

    for (idx, detail) in levels.iter().enumerate() {
      let j = idx + 1;
      if j < self.j_min || j > j_cap {
        continue;
      }
      let n_j = detail.len();
      if n_j < 4 {
        break;
      }
      let mu_j: f64 = detail.iter().map(|d| d * d).sum::<f64>() / n_j as f64;
      if !(mu_j > 0.0 && mu_j.is_finite()) {
        continue;
      }
      let nj_half = (n_j as f64) * 0.5;
      let g_j = (digamma(nj_half) - nj_half.ln()) / LN_2;
      let y_j = mu_j.log2() - g_j;
      let w_j = (n_j as f64) * LN_2 * LN_2 * 0.5;

      log_j.push(j as f64);
      log_mu_corr.push(y_j);
      weights.push(w_j);
    }

    if log_j.len() < 2 {
      return Err(HurstError::NotEnoughScales);
    }

    let (slope, intercept, r_squared) =
      weighted_linreg(&log_j, &log_mu_corr, Some(&weights)).ok_or(HurstError::RegressionFailed)?;
    if !slope.is_finite() {
      return Err(HurstError::RegressionFailed);
    }

    // After (optional) differencing the series is FGN-like with
    // slope = 2H − 1, hence H = (slope + 1) / 2.
    let h = 0.5 * (slope + 1.0);
    let w_sum: f64 = weights.iter().sum();
    let std_err = if w_sum > 0.0 {
      Some((2.0 / w_sum).sqrt() * 0.5)
    } else {
      None
    };

    Ok(HurstResult {
      hurst: T::from_f64_fast(h),
      std_err: std_err.map(T::from_f64_fast),
      n_obs,
      diagnostic: HurstDiagnostic::LogLogRegression {
        slope: T::from_f64_fast(slope),
        intercept: T::from_f64_fast(intercept),
        r_squared: T::from_f64_fast(r_squared),
        log_scales: log_j.into_iter().map(T::from_f64_fast).collect(),
        log_stats: log_mu_corr.into_iter().map(T::from_f64_fast).collect(),
      },
    })
  }
}

/// Quadrature-mirror filter pair from a low-pass scaling filter.
/// Returns `(h, g)` where `g[k] = (-1)^k · h[L-1-k]`.
fn qmf_from_low(h: &[f64]) -> (Vec<f64>, Vec<f64>) {
  let l = h.len();
  let mut g = vec![0.0_f64; l];
  for k in 0..l {
    let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
    g[k] = sign * h[l - 1 - k];
  }
  (h.to_vec(), g)
}

/// Pyramidal DWT with periodic boundary.  Returns one detail-coefficient
/// vector per scale `j = 1..=max_levels` (or until the working vector
/// becomes shorter than the filter).
pub(crate) fn pyramidal_dwt(
  signal: &[f64],
  h_filter: &[f64],
  g_filter: &[f64],
  max_levels: usize,
) -> Vec<Vec<f64>> {
  let l = h_filter.len();
  let mut levels = Vec::with_capacity(max_levels);
  let mut current = signal.to_vec();
  for _ in 0..max_levels {
    let n = current.len();
    if n < l * 2 {
      break;
    }
    let n_half = n / 2;
    let mut low = vec![0.0_f64; n_half];
    let mut detail = vec![0.0_f64; n_half];
    for k in 0..n_half {
      for i in 0..l {
        let idx = (2 * k + i) % n;
        low[k] += h_filter[i] * current[idx];
        detail[k] += g_filter[i] * current[idx];
      }
    }
    levels.push(detail);
    current = low;
  }
  levels
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Unseeded;
  use stochastic_rs_stochastic::noise::fgn::Fgn;
  use stochastic_rs_stochastic::process::fbm::Fbm;

  use super::*;
  use crate::traits::ProcessExt;

  #[test]
  fn wavelet_matches_known_h_on_fbm() {
    let h = 0.7_f64;
    let m = 16;
    let mut acc = 0.0;
    for _ in 0..m {
      let fbm = Fbm::new(h, 8192, Some(1.0), Unseeded);
      let path = fbm.sample();
      let r = Wavelet::default().estimate(path.view()).expect("wavelet");
      acc += r.hurst;
    }
    let h_est = acc / m as f64;
    assert!(
      (h_est - h).abs() < 0.08,
      "Wavelet H(FBM)={h_est:.3}, expected {h:.3}"
    );
  }

  #[test]
  fn wavelet_matches_known_h_on_fgn() {
    let h = 0.3_f64;
    let m = 16;
    let mut acc = 0.0;
    for _ in 0..m {
      let fgn = Fgn::new(h, 8192, Some(1.0), Unseeded);
      let incs = fgn.sample();
      let r = Wavelet {
        take_differences: false,
        ..Default::default()
      }
      .estimate(incs.view())
      .expect("wavelet fgn");
      acc += r.hurst;
    }
    let h_est = acc / m as f64;
    assert!(
      (h_est - h).abs() < 0.10,
      "Wavelet H(FGN)={h_est:.3}, expected {h:.3}"
    );
  }

  #[test]
  fn wavelet_returns_std_err() {
    let fbm = Fbm::new(0.5_f64, 4096, Some(1.0), Unseeded);
    let path = fbm.sample();
    let r = Wavelet::default().estimate(path.view()).expect("wavelet");
    let se = r.std_err.expect("std_err present");
    assert!(se > 0.0 && se < 1.0, "SE={se}");
  }
}
