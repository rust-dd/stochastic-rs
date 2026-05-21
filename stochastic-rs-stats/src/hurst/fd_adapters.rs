//! [`HurstEstimator`] implementations for the fractal-dimension
//! estimators in [`crate::fractal_dim`].
//!
//! Both [`crate::fractal_dim::Higuchi`] and
//! [`crate::fractal_dim::Variogram`] implement
//! [`crate::fractal_dim::FractalDimEstimator`] natively (output: `D`).
//! These adapter `impl` blocks let the same structs be invoked through
//! the [`HurstEstimator`] trait (output: `H = 2 - D`, clamped to
//! `(0, 1)`), so callers who only care about Hurst can use a uniform
//! `HurstEstimator` interface.
//!
//! Disambiguate at the call site when both traits are in scope:
//!
//! ```ignore
//! use stochastic_rs_stats::fractal_dim::{Higuchi, FractalDimEstimator};
//! use stochastic_rs_stats::hurst::HurstEstimator;
//! let est = Higuchi { kmax: 32 };
//! let d_result = FractalDimEstimator::estimate(&est, signal)?;   // → FdResult
//! let h_result = HurstEstimator::estimate(&est, signal)?;        // → HurstResult
//! ```

use ndarray::ArrayView1;

use super::HurstDiagnostic;
use super::HurstError;
use super::HurstEstimator;
use super::HurstResult;
use crate::fractal_dim::FdError;
use crate::fractal_dim::FractalDimEstimator;
use crate::fractal_dim::Higuchi;
use crate::fractal_dim::Variogram;
use crate::traits::FloatExt;

impl<T: FloatExt> HurstEstimator<T> for Higuchi {
  fn estimate(&self, x: ArrayView1<T>) -> Result<HurstResult<T>, HurstError> {
    let fd_result = FractalDimEstimator::<T>::estimate(self, x).map_err(map_fd_err)?;
    fd_to_hurst::<T>(fd_result)
  }
}

impl<T: FloatExt> HurstEstimator<T> for Variogram {
  fn estimate(&self, x: ArrayView1<T>) -> Result<HurstResult<T>, HurstError> {
    let fd_result = FractalDimEstimator::<T>::estimate(self, x).map_err(map_fd_err)?;
    fd_to_hurst::<T>(fd_result)
  }
}

/// Convert a fractal-dimension result into a Hurst result via
/// `H = 2 - D`, clamped to `(0, 1)`.
fn fd_to_hurst<T: FloatExt>(
  fd: crate::fractal_dim::FdResult<T>,
) -> Result<HurstResult<T>, HurstError> {
  let d_f64 = fd.d.to_f64().unwrap_or(f64::NAN);
  if !d_f64.is_finite() {
    return Err(HurstError::DegeneratePath);
  }
  let h = (2.0 - d_f64).clamp(0.0, 1.0);
  Ok(HurstResult {
    hurst: T::from_f64_fast(h),
    std_err: None,
    n_obs: fd.n_obs,
    diagnostic: HurstDiagnostic::FractalDim { d: fd.d },
  })
}

fn map_fd_err(e: FdError) -> HurstError {
  match e {
    FdError::PathTooShort { got, required } => HurstError::TooFewObservations { got, required },
    FdError::NonPositiveP(p) => HurstError::InvalidParameter("p", p),
    FdError::KmaxTooSmall(k) => HurstError::InvalidParameter("kmax", k as f64),
    FdError::DegeneratePath => HurstError::DegeneratePath,
    FdError::NotEnoughScales => HurstError::NotEnoughScales,
    FdError::RegressionFailed => HurstError::RegressionFailed,
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use stochastic_rs_core::simd_rng::Unseeded;
  use stochastic_rs_stochastic::process::fbm::Fbm;

  use super::*;
  use crate::traits::ProcessExt;

  #[test]
  fn higuchi_via_hurst_trait_matches_known_h_on_fbm() {
    let h = 0.7_f64;
    let m = 16;
    let mut acc = 0.0;
    for _ in 0..m {
      let fbm = Fbm::new(h, 4096, Some(1.0), Unseeded);
      let path: Array1<f64> = fbm.sample();
      let r = HurstEstimator::<f64>::estimate(&Higuchi { kmax: 32 }, path.view())
        .expect("Higuchi via HurstEstimator");
      acc += r.hurst;
    }
    let h_est = acc / m as f64;
    assert!(
      (h_est - h).abs() < 0.08,
      "Higuchi(Hurst) H={h_est:.3}, expected {h:.3}"
    );
  }

  #[test]
  fn variogram_via_hurst_trait_matches_known_h_on_fbm() {
    let h = 0.7_f64;
    let m = 16;
    let mut acc = 0.0;
    for _ in 0..m {
      let fbm = Fbm::new(h, 4096, Some(1.0), Unseeded);
      let path: Array1<f64> = fbm.sample();
      let r = HurstEstimator::<f64>::estimate(&Variogram { p: 2.0 }, path.view())
        .expect("Variogram via HurstEstimator");
      acc += r.hurst;
    }
    let h_est = acc / m as f64;
    assert!(
      (h_est - h).abs() < 0.08,
      "Variogram(Hurst) H={h_est:.3}, expected {h:.3}"
    );
  }

  #[test]
  fn higuchi_hurst_diagnostic_carries_d() {
    let fbm = Fbm::new(0.6_f64, 4096, Some(1.0), Unseeded);
    let path: Array1<f64> = fbm.sample();
    let r = HurstEstimator::<f64>::estimate(&Higuchi { kmax: 32 }, path.view())
      .expect("Higuchi via HurstEstimator");
    match r.diagnostic {
      HurstDiagnostic::FractalDim { d } => {
        assert!((1.0..2.0).contains(&d), "D should be in (1, 2), got {d}");
      }
      _ => panic!("expected FractalDim diagnostic"),
    }
  }
}
