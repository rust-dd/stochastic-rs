//! Variogram-ratio (p-power) fractal dimension.
//!
//! $$
//! V_p(l) = \frac{1}{2(N - l)} \sum_{i=l}^{N-1} |X_i - X_{i-l}|^p,
//! \quad
//! D = 2 - \frac{1}{p} \cdot \frac{\log V_p(2) - \log V_p(1)}{\log 2}.
//! $$
//!
//! Reference: Constantine A. G., Hall P. (1994) — *Characterizing surface
//! smoothness via estimation of effective fractal dimension*, JRSS-B 56(1).

use std::f64::consts::LN_2;

use ndarray::ArrayView1;

use super::FdDiagnostic;
use super::FdError;
use super::FdResult;
use super::FractalDimEstimator;
use crate::traits::FloatExt;

/// Variogram-ratio fractal-dimension estimator.
#[derive(Clone, Copy, Debug)]
pub struct Variogram {
  /// p-norm of the variogram (`p > 0`).  Defaults to 2.0.
  pub p: f64,
}

impl Default for Variogram {
  fn default() -> Self {
    Self { p: 2.0 }
  }
}

impl Variogram {
  #[must_use]
  pub fn new(p: f64) -> Self {
    Self { p }
  }
}

impl<T: FloatExt> FractalDimEstimator<T> for Variogram {
  fn estimate(&self, x: ArrayView1<T>) -> Result<FdResult<T>, FdError> {
    let n_obs = x.len();
    if !(self.p.is_finite() && self.p > 0.0) {
      return Err(FdError::NonPositiveP(self.p));
    }
    let p_t = T::from_f64_fast(self.p);
    let (d, v1, v2) = compute_with_diagnostic::<T>(x, p_t)?;
    Ok(FdResult {
      d: T::from_f64_fast(d),
      n_obs,
      diagnostic: FdDiagnostic::VariogramRatio {
        v_short: T::from_f64_fast(v1),
        v_long: T::from_f64_fast(v2),
      },
    })
  }
}

/// Compute the variogram-based fractal dimension of `x`.
///
/// `p` must be strictly positive.  Returns `Err(FdError::PathTooShort)`
/// for `< 3` points, `Err(FdError::NonPositiveP)` for `p ≤ 0`, and
/// `Err(FdError::DegeneratePath)` when the variogram at lag 1 or 2 is
/// non-finite or non-positive (e.g. constant path).
pub fn compute<T: FloatExt>(x: ArrayView1<T>, p: T) -> Result<T, FdError> {
  let (d, _, _) = compute_with_diagnostic::<T>(x, p)?;
  Ok(T::from_f64_fast(d))
}

/// Same as [`compute`] but also returns the variogram-ratio diagnostics
/// `(d, V₁, V₂)`.
pub(crate) fn compute_with_diagnostic<T: FloatExt>(
  x: ArrayView1<T>,
  p: T,
) -> Result<(f64, f64, f64), FdError> {
  if x.len() < 3 {
    return Err(FdError::PathTooShort {
      got: x.len(),
      required: 3,
    });
  }

  let p_f64 = p.to_f64().unwrap_or(f64::NAN);
  if !(p_f64 > 0.0 && p_f64.is_finite()) {
    return Err(FdError::NonPositiveP(p_f64));
  }

  let sum1: f64 = (1..x.len())
    .map(|i| {
      let d = x[i].to_f64().unwrap_or(f64::NAN) - x[i - 1].to_f64().unwrap_or(f64::NAN);
      d.abs().powf(p_f64)
    })
    .sum();
  let sum2: f64 = (2..x.len())
    .map(|i| {
      let d = x[i].to_f64().unwrap_or(f64::NAN) - x[i - 2].to_f64().unwrap_or(f64::NAN);
      d.abs().powf(p_f64)
    })
    .sum();

  let n = x.len();
  let v1 = sum1 / (2.0 * (n - 1) as f64);
  let v2 = sum2 / (2.0 * (n - 2) as f64);
  if !(v1.is_finite() && v2.is_finite() && v1 > 0.0 && v2 > 0.0) {
    return Err(FdError::DegeneratePath);
  }

  let d = 2.0 - (1.0 / p_f64) * ((v2.ln() - v1.ln()) / LN_2);
  Ok((d, v1, v2))
}
