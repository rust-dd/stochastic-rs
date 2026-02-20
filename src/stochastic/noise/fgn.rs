//! # fGN
//!
//! $$
//! \operatorname{Cov}(\Delta B_i^H,\Delta B_j^H)=\tfrac12\left(|k+1|^{2H}-2|k|^{2H}+|k-1|^{2H}\right),\ k=i-j
//! $$
//!
mod core;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "python")]
mod python;

pub use core::FGN;

#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use either::Either;
use ndarray::Array1;
#[cfg(feature = "cuda")]
use ndarray::Array2;
#[cfg(feature = "python")]
pub use python::PyFGN;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

impl<T: FloatExt> ProcessExt<T> for FGN<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    self.sample_cpu()
  }

  #[cfg(feature = "cuda")]
  fn sample_cuda(&self, m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    self.sample_cuda_impl(m)
  }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
  use anyhow::Result;
  use either::Either;

  use super::FGN;
  use crate::traits::ProcessExt;

  fn sample_cuda_paths(
    h: f32,
    n: usize,
    t: f32,
    m_total: usize,
    chunk_size: usize,
  ) -> Result<Vec<Vec<f64>>> {
    let fgn = FGN::<f32>::new(h, n, Some(t));
    let mut out = Vec::with_capacity(m_total);

    while out.len() < m_total {
      let m = (m_total - out.len()).min(chunk_size.max(1));
      match fgn.sample_cuda(m)? {
        Either::Left(path) => {
          out.push(path.iter().map(|&x| x as f64).collect());
        }
        Either::Right(paths) => {
          for row in paths.outer_iter() {
            out.push(row.iter().map(|&x| x as f64).collect());
          }
        }
      }
    }

    Ok(out)
  }

  fn unit_lag_covariance(h: f64, k: usize) -> f64 {
    if k == 0 {
      1.0
    } else {
      0.5
        * (((k + 1) as f64).powf(2.0 * h) - 2.0 * (k as f64).powf(2.0 * h)
          + ((k - 1) as f64).powf(2.0 * h))
    }
  }

  fn lag_covariance(paths: &[Vec<f64>], mean: f64, lag: usize) -> f64 {
    let mut s = 0.0;
    let mut c = 0usize;
    for p in paths {
      for i in 0..(p.len() - lag) {
        s += (p[i] - mean) * (p[i + lag] - mean);
        c += 1;
      }
    }
    s / c as f64
  }

  #[test]
  fn fgn_cuda_marginal_distribution_and_covariance_match_theory() {
    let h = 0.72_f32;
    let n = 1024_usize;
    let t = 1.0_f32;
    let m = 512_usize;
    let paths = match sample_cuda_paths(h, n, t, m, 128) {
      Ok(paths) => paths,
      Err(err) => {
        eprintln!("Skipping CUDA FGN distribution test: {err}");
        return;
      }
    };

    let mut values = Vec::with_capacity(m * n);
    for p in &paths {
      values.extend_from_slice(p);
    }

    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let var = values
      .iter()
      .map(|x| {
        let d = *x - mean;
        d * d
      })
      .sum::<f64>()
      / count;

    let h64 = h as f64;
    let dt = (t as f64) / n as f64;
    let var_theory = dt.powf(2.0 * h64);
    let cov1_theory = var_theory * unit_lag_covariance(h64, 1);
    let cov4_theory = var_theory * unit_lag_covariance(h64, 4);
    let cov1_emp = lag_covariance(&paths, mean, 1);
    let cov4_emp = lag_covariance(&paths, mean, 4);

    assert!(mean.abs() < 8e-4, "mean too far from zero: {mean}");
    assert!(
      ((var / var_theory) - 1.0).abs() < 0.10,
      "variance mismatch: emp={var}, theory={var_theory}"
    );
    assert!(
      ((cov1_emp / cov1_theory) - 1.0).abs() < 0.16,
      "lag-1 covariance mismatch: emp={cov1_emp}, theory={cov1_theory}"
    );
    assert!(
      ((cov4_emp / cov4_theory) - 1.0).abs() < 0.20,
      "lag-4 covariance mismatch: emp={cov4_emp}, theory={cov4_theory}"
    );
  }

  #[test]
  fn fgn_cuda_lag1_correlation_sign_matches_hurst_regime() {
    let n = 1024_usize;
    let t = 1.0_f32;
    let m = 320_usize;

    let low_paths = match sample_cuda_paths(0.25, n, t, m, 128) {
      Ok(paths) => paths,
      Err(err) => {
        eprintln!("Skipping CUDA FGN lag-sign test (low-h): {err}");
        return;
      }
    };
    let high_paths = match sample_cuda_paths(0.80, n, t, m, 128) {
      Ok(paths) => paths,
      Err(err) => {
        eprintln!("Skipping CUDA FGN lag-sign test (high-h): {err}");
        return;
      }
    };

    let low_vals: Vec<f64> = low_paths.iter().flatten().copied().collect();
    let high_vals: Vec<f64> = high_paths.iter().flatten().copied().collect();
    let low_mean = low_vals.iter().sum::<f64>() / low_vals.len() as f64;
    let high_mean = high_vals.iter().sum::<f64>() / high_vals.len() as f64;
    let low_var = low_vals
      .iter()
      .map(|x| {
        let d = *x - low_mean;
        d * d
      })
      .sum::<f64>()
      / low_vals.len() as f64;
    let high_var = high_vals
      .iter()
      .map(|x| {
        let d = *x - high_mean;
        d * d
      })
      .sum::<f64>()
      / high_vals.len() as f64;

    let low_rho1 = lag_covariance(&low_paths, low_mean, 1) / low_var;
    let high_rho1 = lag_covariance(&high_paths, high_mean, 1) / high_var;

    assert!(
      low_rho1 < -0.08,
      "expected negative lag-1 correlation for H<0.5, got {low_rho1}"
    );
    assert!(
      high_rho1 > 0.08,
      "expected positive lag-1 correlation for H>0.5, got {high_rho1}"
    );
  }
}
