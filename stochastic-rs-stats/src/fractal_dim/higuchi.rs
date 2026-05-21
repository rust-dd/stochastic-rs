//! Higuchi (1988) curve-length fractal dimension.
//!
//! Reference: Higuchi T. (1988) — *Approach to an irregular time series on
//! the basis of the fractal theory*, Physica D 31(2), 277–283,
//! DOI: 10.1016/0167-2789(88)90081-4.

use linreg::linear_regression;
use ndarray::Array1;
use ndarray::ArrayView1;

use super::FdDiagnostic;
use super::FdError;
use super::FdResult;
use super::FractalDimEstimator;
use crate::traits::FloatExt;

/// Higuchi fractal-dimension estimator.
#[derive(Clone, Copy, Debug)]
pub struct Higuchi {
  /// Maximum `k` in the Higuchi curve-length method.
  pub kmax: usize,
}

impl Default for Higuchi {
  fn default() -> Self {
    Self { kmax: 32 }
  }
}

impl<T: FloatExt> FractalDimEstimator<T> for Higuchi {
  fn estimate(&self, x: ArrayView1<T>) -> Result<FdResult<T>, FdError> {
    let n_obs = x.len();
    let (d, intercept, r_squared, log_scales, log_stats) =
      compute_with_diagnostic::<T>(x, self.kmax)?;
    Ok(FdResult {
      // Higuchi's `D` is the regression slope itself.
      d: T::from_f64_fast(d),
      n_obs,
      diagnostic: FdDiagnostic::LogLogRegression {
        slope: T::from_f64_fast(d),
        intercept: T::from_f64_fast(intercept),
        r_squared: T::from_f64_fast(r_squared),
        log_scales: log_scales.into_iter().map(T::from_f64_fast).collect(),
        log_stats: log_stats.into_iter().map(T::from_f64_fast).collect(),
      },
    })
  }
}

/// Compute the Higuchi fractal dimension of `x`.
///
/// $$
/// L_m(k) = \frac{N - 1}{k \cdot n_{max}} \cdot \frac{1}{k}
///          \sum_{j=1}^{n_{max}} |X_{m+jk} - X_{m+(j-1)k}|,
/// \quad
/// L(k) = \frac{1}{k} \sum_{m=1}^{k} L_m(k),
/// $$
///
/// and the dimension is the slope of `log L(k)` against `log(1/k)`.
///
/// Returns `Err(FdError::PathTooShort)` for `< 3` points,
/// `Err(FdError::KmaxTooSmall)` for `kmax < 2`,
/// `Err(FdError::NotEnoughScales)` when fewer than two valid scales
/// survive the finiteness filter, and `Err(FdError::RegressionFailed)`
/// if the underlying linear regression cannot be solved.
pub fn compute<T: FloatExt>(x: ArrayView1<T>, kmax: usize) -> Result<T, FdError> {
  let (d, _, _, _, _) = compute_with_diagnostic::<T>(x, kmax)?;
  Ok(T::from_f64_fast(d))
}

/// Same as [`compute`] but also returns the log-log regression
/// diagnostics `(d, intercept, r_squared, log_scales, log_stats)`.
/// For Higuchi `d == slope`.
pub(crate) fn compute_with_diagnostic<T: FloatExt>(
  x: ArrayView1<T>,
  kmax: usize,
) -> Result<(f64, f64, f64, Vec<f64>, Vec<f64>), FdError> {
  let n_times = x.len();
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
    let mut lm_sum = 0.0_f64;
    let mut lm_count = 0usize;

    for m in 0..k {
      let n_max = (n_times - m - 1) / k;
      if n_max == 0 {
        continue;
      }

      let mut ll = 0.0_f64;
      for j in 1..=n_max {
        let a = x[m + j * k].to_f64().unwrap_or(f64::NAN);
        let b = x[m + (j - 1) * k].to_f64().unwrap_or(f64::NAN);
        ll += (a - b).abs();
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
        x_reg[used] = (1.0_f64 / k as f64).ln();
        y_reg[used] = lk.ln();
        used += 1;
      }
    }
  }

  if used < 2 {
    return Err(FdError::NotEnoughScales);
  }
  let x_slice = &x_reg.as_slice().unwrap()[..used];
  let y_slice = &y_reg.as_slice().unwrap()[..used];
  let (slope, intercept) =
    linear_regression(x_slice, y_slice).map_err(|_| FdError::RegressionFailed)?;

  let log_scales: Vec<f64> = x_slice.to_vec();
  let log_stats: Vec<f64> = y_slice.to_vec();

  let x_mean = log_scales.iter().sum::<f64>() / used as f64;
  let y_mean = log_stats.iter().sum::<f64>() / used as f64;
  let mut sxy = 0.0;
  let mut sxx = 0.0;
  let mut syy = 0.0;
  for i in 0..used {
    let dx = log_scales[i] - x_mean;
    let dy = log_stats[i] - y_mean;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  let r_squared = if sxx > 0.0 && syy > 0.0 {
    (sxy * sxy) / (sxx * syy)
  } else {
    0.0
  };

  Ok((slope, intercept, r_squared, log_scales, log_stats))
}
