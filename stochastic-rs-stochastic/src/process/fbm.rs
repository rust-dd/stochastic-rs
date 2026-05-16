//! # fBM
//!
//! $$
//! \mathbb E[B_t^H B_s^H]=\tfrac12\left(t^{2H}+s^{2H}-|t-s|^{2H}\right)
//! $$
//!
#[cfg(any(
  feature = "gpu",
  feature = "cuda-native",
  feature = "accelerate",
  feature = "metal"
))]
use anyhow::Result;
#[cfg(any(
  feature = "gpu",
  feature = "cuda-native",
  feature = "accelerate",
  feature = "metal"
))]
use either::Either;
use ndarray::Array1;
#[cfg(any(
  feature = "gpu",
  feature = "cuda-native",
  feature = "accelerate",
  feature = "metal",
  feature = "python"
))]
use ndarray::Array2;
#[cfg(feature = "python")]
use numpy::IntoPyArray;
#[cfg(feature = "python")]
use numpy::PyArray1;
#[cfg(feature = "python")]
use numpy::PyArray2;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::noise::fgn::Fgn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct Fbm<T: FloatExt, S: SeedExt = Unseeded> {
  /// Hurst parameter (`0 < H < 1`) controlling roughness and memory.
  pub hurst: T,
  /// Number of discrete time points in the generated path.
  pub n: usize,
  /// Total simulation horizon (defaults to `1` if `None`).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
  fgn: Fgn<T>,
}

impl<T: FloatExt, S: SeedExt> Fbm<T, S> {
  pub fn new(hurst: T, n: usize, t: Option<T>, seed: S) -> Self {
    assert!(n >= 2, "n must be at least 2");

    Self {
      hurst,
      n,
      t,
      seed,
      fgn: Fgn::new(hurst, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Fbm<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let fgn = self.fgn.sample_cpu_impl(&self.seed.derive());
    let mut fbm = Array1::<T>::zeros(self.n);

    for i in 1..self.n {
      fbm[i] = fbm[i - 1] + fgn[i - 1];
    }

    fbm
  }

  #[cfg(feature = "gpu")]
  fn sample_gpu(&self, m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    Self::fgn_to_fbm(self.n, self.fgn.sample_gpu(m)?)
  }

  #[cfg(feature = "cuda-native")]
  fn sample_cuda_native(&self, m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    Self::fgn_to_fbm(self.n, self.fgn.sample_cuda_native(m)?)
  }

  #[cfg(feature = "accelerate")]
  fn sample_accelerate(&self, m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    Self::fgn_to_fbm(self.n, self.fgn.sample_accelerate(m)?)
  }

  #[cfg(feature = "metal")]
  fn sample_metal(&self, m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    Self::fgn_to_fbm(self.n, self.fgn.sample_metal(m)?)
  }
}

#[cfg(any(
  feature = "gpu",
  feature = "cuda-native",
  feature = "accelerate",
  feature = "metal"
))]
impl<T: FloatExt, S: SeedExt> Fbm<T, S> {
  fn fgn_to_fbm(
    n: usize,
    fgn_out: Either<Array1<T>, Array2<T>>,
  ) -> Result<Either<Array1<T>, Array2<T>>> {
    match fgn_out {
      Either::Left(fgn_path) => {
        let mut fbm = Array1::<T>::zeros(n);
        for i in 1..n {
          fbm[i] = fbm[i - 1] + fgn_path[i - 1];
        }
        Ok(Either::Left(fbm))
      }
      Either::Right(fgn_paths) => {
        let rows = fgn_paths.nrows();
        let mut fbm_paths = Array2::<T>::zeros((rows, n));
        for r in 0..rows {
          for i in 1..n {
            fbm_paths[[r, i]] = fbm_paths[[r, i - 1]] + fgn_paths[[r, i - 1]];
          }
        }
        Ok(Either::Right(fbm_paths))
      }
    }
  }
}

impl<T: FloatExt, S: SeedExt> Fbm<T, S> {
  /// Calculate the Malliavin derivative
  ///
  /// The Malliavin derivative of the fractional Brownian motion is given by:
  /// D_s B^H_t = 1 / Γ(H + 1/2) (t - s)^{H - 1/2}
  ///
  /// where B^H_t is the fractional Brownian motion with Hurst parameter H in Mandelbrot-Van Ness representation as
  /// B^H_t = 1 / Γ(H + 1/2) ∫_0^t (t - s)^{H - 1/2} dW_s
  /// which is a truncated Wiener integral.
  pub fn malliavin(&self) -> Array1<T> {
    let dt = self.fgn.dt();
    let mut m = Array1::zeros(self.n);
    let g = stochastic_rs_distributions::special::gamma(self.hurst.to_f64().unwrap() + 0.5);

    for i in 0..self.n {
      m[i] = T::one() / T::from_f64_fast(g)
        * (T::from_usize_(i) * dt).powf(self.hurst - T::from_f64_fast(0.5));
    }

    m
  }
}

#[cfg(feature = "python")]
#[pyclass]
pub struct PyFbm {
  inner: Option<Fbm<f64>>,
  seeded: Option<Fbm<f64, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyFbm {
  #[new]
  #[pyo3(signature = (hurst, n, t=None, seed=None))]
  fn new(hurst: f64, n: usize, t: Option<f64>, seed: Option<u64>) -> Self {
    match seed {
      Some(s) => Self {
        inner: None,
        seeded: Some(Fbm::new(hurst, n, t, Deterministic::new(s))),
      },
      None => Self {
        inner: Some(Fbm::new(hurst, n, t, Unseeded)),
        seeded: None,
      },
    }
  }

  fn sample<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    py_dispatch_f64!(self, |inner| inner.sample().into_pyarray(py))
  }

  fn sample_par<'py>(&self, py: Python<'py>, m: usize) -> Bound<'py, PyArray2<f64>> {
    py_dispatch_f64!(self, |inner| {
      let paths = inner.sample_par(m);
      let n = paths[0].len();
      let mut result = Array2::<f64>::zeros((m, n));
      for (i, path) in paths.iter().enumerate() {
        result.row_mut(i).assign(path);
      }
      result.into_pyarray(py)
    })
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::special::erf;

  use super::*;

  fn nearest_quantile(sorted: &[f64], p: f64) -> f64 {
    let idx = (((sorted.len() - 1) as f64) * p).round() as usize;
    sorted[idx]
  }

  fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
  }

  fn regression_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let x_mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let y_mean = ys.iter().sum::<f64>() / ys.len() as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
      num += (x - x_mean) * (y - y_mean);
      den += (x - x_mean) * (x - x_mean);
    }
    num / den
  }

  #[test]
  fn fbm_terminal_marginal_is_gaussian_with_correct_scale() {
    let h = 0.72_f64;
    let t = 1.0_f64;
    let n = 2048_usize;
    let m = 6000_usize;
    // Seeded `Fbm::sample(&self)` advances the internal atomic seed state
    // each call, so a single instance yields `m` independent paths
    // deterministically.
    let fbm = Fbm::new(h, n, Some(t), Deterministic::new(0xFBC0_FFEE_u64));

    let mut endpoints = Vec::with_capacity(m);
    for _ in 0..m {
      let x = fbm.sample();
      endpoints.push(x[n - 1]);
    }

    let mean = endpoints.iter().sum::<f64>() / m as f64;
    let var = endpoints
      .iter()
      .map(|x| {
        let d = *x - mean;
        d * d
      })
      .sum::<f64>()
      / m as f64;
    let std = var.sqrt();
    let var_theory = t.powf(2.0 * h);

    let mut sorted = endpoints.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q025 = (nearest_quantile(&sorted, 0.025) - mean) / std;
    let q975 = (nearest_quantile(&sorted, 0.975) - mean) / std;
    let mut ks = 0.0_f64;
    for (i, x) in sorted.iter().enumerate() {
      let z = (*x - mean) / std;
      let f = standard_normal_cdf(z);
      let e1 = ((i + 1) as f64 / m as f64 - f).abs();
      let e2 = (i as f64 / m as f64 - f).abs();
      ks = ks.max(e1.max(e2));
    }

    assert!(mean.abs() < 0.05, "terminal mean too far from 0: {mean}");
    assert!(
      ((var / var_theory) - 1.0).abs() < 0.05,
      "terminal variance mismatch: emp={var}, theory={var_theory}"
    );
    assert!(
      (q025 + 1.96).abs() < 0.1 && (q975 - 1.96).abs() < 0.1,
      "terminal quantile mismatch: q025={q025}, q975={q975}"
    );
    assert!(ks < 0.05, "KS distance too large: {ks}");
  }

  #[test]
  fn fbm_covariance_kernel_matches_theory() {
    let h = 0.72_f64;
    let t_max = 1.0_f64;
    let n = 2048_usize;
    let m = 5000_usize;
    let fbm = Fbm::new(h, n, Some(t_max), Unseeded);
    let dt = t_max / (n as f64 - 1.0);
    let idxs = [n / 4, n / 2, 3 * n / 4, n - 1];

    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(m); idxs.len()];
    for _ in 0..m {
      let path = fbm.sample();
      for (j, &idx) in idxs.iter().enumerate() {
        samples[j].push(path[idx]);
      }
    }

    let means: Vec<f64> = samples
      .iter()
      .map(|v| v.iter().sum::<f64>() / v.len() as f64)
      .collect();

    let mut off_diag_rel_sum = 0.0_f64;
    let mut off_diag_count = 0usize;

    for i in 0..idxs.len() {
      for j in i..idxs.len() {
        let mut cov = 0.0;
        for k in 0..m {
          cov += (samples[i][k] - means[i]) * (samples[j][k] - means[j]);
        }
        cov /= m as f64;

        let ti = idxs[i] as f64 * dt;
        let tj = idxs[j] as f64 * dt;
        let cov_theory =
          0.5 * (ti.powf(2.0 * h) + tj.powf(2.0 * h) - (ti - tj).abs().powf(2.0 * h));
        let rel_err = ((cov / cov_theory) - 1.0).abs();
        if i == j {
          assert!(
            rel_err < 0.08,
            "variance mismatch at ({i},{j}): emp={cov}, theory={cov_theory}, rel_err={rel_err}"
          );
        } else {
          off_diag_rel_sum += rel_err;
          off_diag_count += 1;
        }
      }
    }

    let off_diag_mean_rel_err = off_diag_rel_sum / off_diag_count as f64;
    assert!(
      off_diag_mean_rel_err < 0.08,
      "off-diagonal mean relative covariance error too large: {off_diag_mean_rel_err}"
    );
  }

  #[test]
  fn fbm_hurst_scaling_matches_theory() {
    let h = 0.72_f64;
    let t_max = 1.0_f64;
    let n = 2048_usize;
    let m = 2200_usize;
    let fbm = Fbm::new(h, n, Some(t_max), Unseeded);
    let dt = t_max / (n as f64 - 1.0);
    let idxs = [n / 16, n / 8, n / 4, n / 2, n - 1];

    let mut buckets: Vec<Vec<f64>> = vec![Vec::with_capacity(m); idxs.len()];
    for _ in 0..m {
      let path = fbm.sample();
      for (j, &idx) in idxs.iter().enumerate() {
        buckets[j].push(path[idx]);
      }
    }

    let mut xs = Vec::with_capacity(idxs.len());
    let mut ys = Vec::with_capacity(idxs.len());
    for (j, &idx) in idxs.iter().enumerate() {
      let vals = &buckets[j];
      let mean = vals.iter().sum::<f64>() / vals.len() as f64;
      let var = vals
        .iter()
        .map(|x| {
          let d = *x - mean;
          d * d
        })
        .sum::<f64>()
        / vals.len() as f64;
      xs.push((idx as f64 * dt).ln());
      ys.push(var.ln());
    }

    let h_est = 0.5 * regression_slope(&xs, &ys);
    assert!(
      (h_est - h).abs() < 0.05,
      "hurst mismatch from scaling: h_est={h_est}, h={h}"
    );
  }

  // `fbm_fractal_dimension_matches_theory` lives in
  // `stochastic-rs-stats/tests/fractal_dim_validation.rs` because it exercises
  // the `FractalDim` estimator from the stats crate.
}
