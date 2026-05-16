//! # Fd
//!
//! $$
//! D=2-\frac{1}{p}\frac{\log V_p(2)-\log V_p(1)}{\log 2}
//! $$
//!
use std::f64::consts::LN_2;
use std::fmt;

use linreg::linear_regression;
use ndarray::Array1;

/// Errors returned by [`FractalDim`] estimators.
///
/// Replaces the `panic!()` failure modes of the v1 API; estimators now
/// surface degenerate / non-finite path conditions through this enum and
/// callers can decide whether to fall back, return NaN, or propagate.
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

/// Fractal dimension.
pub struct FractalDim {
  /// Observed data/sample vector.
  pub x: Array1<f64>,
}

impl FractalDim {
  #[must_use]
  pub fn new(x: Array1<f64>) -> Self {
    Self { x }
  }

  /// Calculate the variogram-based fractal dimension of the path.
  ///
  /// Returns `Err(FdError::PathTooShort)` for paths with `< 3` points,
  /// `Err(FdError::NonPositiveP)` for `p ≤ 0`, and
  /// `Err(FdError::DegeneratePath)` when the variogram is undefined
  /// (e.g. constant path).
  pub fn variogram(&self, p: Option<f64>) -> Result<f64, FdError> {
    if self.x.len() < 3 {
      return Err(FdError::PathTooShort {
        got: self.x.len(),
        required: 3,
      });
    }

    let p = p.unwrap_or(1.0);
    if p <= 0.0 {
      return Err(FdError::NonPositiveP(p));
    }
    let sum1: f64 = (1..self.x.len())
      .map(|i| (self.x[i] - self.x[i - 1]).abs().powf(p))
      .sum();
    let sum2: f64 = (2..self.x.len())
      .map(|i| (self.x[i] - self.x[i - 2]).abs().powf(p))
      .sum();

    let vp = |increments: f64, l: usize, x_len: usize| -> f64 {
      1.0 / (2.0 * (x_len - l) as f64) * increments
    };

    let v1 = vp(sum1, 1, self.x.len());
    let v2 = vp(sum2, 2, self.x.len());
    if v1 <= 0.0 || v2 <= 0.0 || !v1.is_finite() || !v2.is_finite() {
      return Err(FdError::DegeneratePath);
    }

    Ok(2.0 - (1.0 / p) * ((v2.ln() - v1.ln()) / LN_2))
  }

  /// Calculate the Higuchi fractal dimension of the path.
  ///
  /// Returns `Err(FdError::PathTooShort)` for `< 3` points,
  /// `Err(FdError::KmaxTooSmall)` for `kmax < 2`,
  /// `Err(FdError::NotEnoughScales)` when fewer than two valid scales
  /// survive the finiteness filter, and `Err(FdError::RegressionFailed)`
  /// if the underlying linear regression cannot be solved.
  pub fn higuchi_fd(&self, kmax: usize) -> Result<f64, FdError> {
    let n_times = self.x.len();
    if n_times < 3 {
      return Err(FdError::PathTooShort {
        got: n_times,
        required: 3,
      });
    }
    if kmax < 2 {
      return Err(FdError::KmaxTooSmall(kmax));
    }

    let k_upper = kmax.min(n_times - 1);
    let mut x_reg = Array1::<f64>::zeros(k_upper);
    let mut y_reg = Array1::<f64>::zeros(k_upper);
    let mut used = 0usize;

    for k in 1..=k_upper {
      let mut lm_sum = 0.0;
      let mut lm_count = 0usize;

      for m in 0..k {
        let n_max = (n_times - m - 1) / k;
        if n_max == 0 {
          continue;
        }

        let mut ll = 0.0;
        for j in 1..=n_max {
          ll += (self.x[m + j * k] - self.x[m + (j - 1) * k]).abs();
        }

        ll /= k as f64;
        ll *= (n_times - 1) as f64 / (k * n_max) as f64;
        if ll.is_finite() && ll > 0.0 {
          lm_sum += ll;
          lm_count += 1;
        }
      }

      if lm_count > 0 {
        let lk = lm_sum / lm_count as f64;
        if lk.is_finite() && lk > 0.0 {
          x_reg[used] = (1.0 / k as f64).ln();
          y_reg[used] = lk.ln();
          used += 1;
        }
      }
    }

    if used < 2 {
      return Err(FdError::NotEnoughScales);
    }
    let x = &x_reg.as_slice().unwrap()[..used];
    let y = &y_reg.as_slice().unwrap()[..used];
    let (slope, _) = linear_regression(x, y).map_err(|_| FdError::RegressionFailed)?;
    Ok(slope)
  }
}

/// Estimate the Hurst exponent from a close-price series.
///
/// Uses Higuchi fractal dimension on a realized-volatility proxy
/// (rolling mean absolute return), cross-validated against an
/// absolute-return based estimate.
///
/// Returns H in \[0.05, 0.45\].  Falls back to 0.1 when data is
/// insufficient or the estimate is unreliable.
///
/// # Arguments
/// * `closes` — Daily (or intraday) close prices as `ArrayView1`,
///   length >= 30.
pub fn estimate_hurst(closes: ndarray::ArrayView1<f64>) -> f64 {
  let n = closes.len();
  if n < 30 {
    return 0.1;
  }

  let rets: Vec<f64> = (1..n)
    .filter_map(|i| {
      let r = (closes[i] / closes[i - 1]).ln();
      if r.is_finite() { Some(r) } else { None }
    })
    .collect();

  if rets.len() < 30 {
    return 0.1;
  }

  // Realized vol proxy: rolling mean absolute return
  let window = 5.min(rets.len() / 4).max(2);
  let vol_proxy: Vec<f64> = rets
    .windows(window)
    .map(|w| {
      let sum: f64 = w.iter().map(|r| r.abs()).sum();
      sum / window as f64
    })
    .filter(|&v| v.is_finite() && v > 0.0)
    .collect();

  if vol_proxy.len() < 20 {
    let abs_rets: Array1<f64> = Array1::from_vec(
      rets
        .iter()
        .map(|r| r.abs())
        .filter(|&r| r.is_finite() && r > 0.0)
        .collect(),
    );
    return hurst_from_signal(abs_rets.view());
  }

  let vol_arr = Array1::from_vec(vol_proxy);
  let h_rv = hurst_from_signal(vol_arr.view());

  let abs_arr = Array1::from_vec(
    rets
      .iter()
      .map(|r| r.abs())
      .filter(|&r| r.is_finite() && r > 0.0)
      .collect(),
  );
  let h_abs = hurst_from_signal(abs_arr.view());

  // Cross-validate: if estimates disagree by > 0.15, use the lower (conservative)
  if (h_rv - h_abs).abs() > 0.15 {
    h_rv.min(h_abs).clamp(0.05, 0.45)
  } else {
    (0.65 * h_rv + 0.35 * h_abs).clamp(0.05, 0.45)
  }
}

/// Estimate Hurst exponent from an arbitrary positive signal via Higuchi FD.
///
/// Converts fractal dimension D to H = 2 − D, clamped to \[0.05, 0.45\].
/// Returns 0.1 on degenerate input or when the underlying Higuchi
/// estimator returns [`FdError`] (no panic / unwind required since
/// `higuchi_fd` is now [`Result`]-based).
///
/// # Arguments
/// * `signal` — Positive-valued time series as `ArrayView1`.
pub fn hurst_from_signal(signal: ndarray::ArrayView1<f64>) -> f64 {
  if signal.len() < 20 {
    return 0.1;
  }
  let owned = signal.to_owned();
  let kmax = 64.min(signal.len() / 4).max(4);

  let fd = FractalDim::new(owned);
  match fd.higuchi_fd(kmax) {
    Ok(d) if d.is_finite() && d > 1.0 && d < 2.0 => (2.0 - d).clamp(0.05, 0.45),
    _ => 0.1,
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use stochastic_rs_core::simd_rng::Unseeded;
  use stochastic_rs_stochastic::process::fbm::Fbm;

  use super::FdError;
  use super::FractalDim;
  use crate::traits::ProcessExt;

  #[test]
  fn variogram_fbm_matches_theory_on_average() {
    let h = 0.72_f64;
    let d_theory = 2.0 - h;
    let n = 4096_usize;
    let m = 192_usize;
    let fbm = Fbm::new(h, n, Some(1.0), Unseeded);

    let mut d_sum = 0.0;
    for _ in 0..m {
      let x = fbm.sample();
      let fd = FractalDim::new(x);
      d_sum += fd
        .variogram(Some(2.0))
        .expect("variogram should succeed on fBM path");
    }
    let d_est = d_sum / m as f64;
    assert!(
      (d_est - d_theory).abs() < 0.05,
      "variogram FD mismatch: D_est={d_est}, D={d_theory}"
    );
  }

  #[test]
  fn higuchi_fbm_matches_theory_on_average() {
    let h = 0.72_f64;
    let d_theory = 2.0 - h;
    let n = 4096_usize;
    let kmax = 32_usize;
    let m = 96_usize;
    let fbm = Fbm::new(h, n, Some(1.0), Unseeded);

    let mut d_sum = 0.0;
    for _ in 0..m {
      let x = fbm.sample();
      let fd = FractalDim::new(x);
      d_sum += fd
        .higuchi_fd(kmax)
        .expect("Higuchi should succeed on fBM path");
    }
    let d_est = d_sum / m as f64;
    assert!(
      (d_est - d_theory).abs() < 0.05,
      "Higuchi FD mismatch: D_est={d_est}, D={d_theory}"
    );
  }

  #[test]
  fn variogram_rejects_degenerate_path() {
    let x = Array1::from_elem(128, 1.0);
    let fd = FractalDim::new(x);
    assert_eq!(fd.variogram(Some(2.0)), Err(FdError::DegeneratePath));
  }

  #[test]
  fn variogram_rejects_short_path() {
    let x = Array1::from_vec(vec![1.0, 2.0]);
    let fd = FractalDim::new(x);
    match fd.variogram(None) {
      Err(FdError::PathTooShort {
        got: 2,
        required: 3,
      }) => {}
      other => panic!("expected PathTooShort, got {:?}", other),
    }
  }

  #[test]
  fn variogram_rejects_non_positive_p() {
    let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let fd = FractalDim::new(x);
    assert_eq!(fd.variogram(Some(0.0)), Err(FdError::NonPositiveP(0.0)));
    assert_eq!(fd.variogram(Some(-1.0)), Err(FdError::NonPositiveP(-1.0)));
  }

  #[test]
  fn higuchi_rejects_kmax_too_small() {
    let x = Array1::from_vec((0..100).map(|i| i as f64).collect());
    let fd = FractalDim::new(x);
    assert_eq!(fd.higuchi_fd(0), Err(FdError::KmaxTooSmall(0)));
    assert_eq!(fd.higuchi_fd(1), Err(FdError::KmaxTooSmall(1)));
  }
}
