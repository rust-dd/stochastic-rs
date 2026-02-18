//! # Core
//!
//! $$
//! \operatorname{Cov}(\Delta B_i^H,\Delta B_j^H)=\tfrac12\left(|k+1|^{2H}-2|k|^{2H}+|k-1|^{2H}\right),\ k=i-j
//! $$
//!
use std::sync::Arc;

use ndarray::concatenate;
use ndarray::prelude::*;
use ndrustfft::FftHandler;
use ndrustfft::ndfft_inplace_par;
use ndrustfft::ndfft_par;
use num_complex::Complex;

use crate::traits::FloatExt;

pub struct FGN<T: FloatExt> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
  /// Internal FFT length (power-of-two padded).
  pub n: usize,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Model parameter controlling process dynamics.
  pub offset: usize,
  pub(crate) out_len: usize,
  pub(crate) scale: T,
  /// Model parameter controlling process dynamics.
  pub sqrt_eigenvalues: Arc<Array1<T>>,
  /// Model parameter controlling process dynamics.
  pub fft_handler: Arc<FftHandler<T>>,
}

impl<T: FloatExt> FGN<T> {
  pub fn dt(&self) -> T {
    let step_count = self.out_len.max(1);
    self.t.unwrap_or(T::one()) / T::from_usize_(step_count)
  }

  #[must_use]
  pub fn new(hurst: T, n: usize, t: Option<T>) -> Self {
    if !(T::zero()..=T::one()).contains(&hurst) {
      panic!("Hurst parameter must be between 0 and 1");
    }

    let offset = n.next_power_of_two() - n;
    let out_len = n;
    let n = n.next_power_of_two();
    let mut r = Array1::linspace(T::zero(), T::from_usize_(n), n + 1);
    let f2 = T::from_usize_(2);
    r.mapv_inplace(|x| {
      if x == T::zero() {
        T::one()
      } else {
        T::from_f64_fast(0.5)
          * ((x + T::one()).powf(f2 * hurst) - f2 * x.powf(f2 * hurst)
            + (x - T::one()).powf(f2 * hurst))
      }
    });
    let r = concatenate(
      Axis(0),
      #[allow(clippy::reversed_empty_ranges)]
      &[r.view(), r.slice(s![..;-1]).slice(s![1..-1]).view()],
    )
    .unwrap();
    let data = r.mapv(|v| Complex::new(v, T::zero()));
    let r_fft = FftHandler::new(r.len());
    let mut eig_fft = Array1::<Complex<T>>::zeros(r.len());
    ndfft_par(&data, &mut eig_fft, &r_fft, 0);
    let norm = T::from_usize_(2 * n);
    let mut sqrt_eigenvalues = Array1::<T>::zeros(r.len());
    for (dst, eig) in sqrt_eigenvalues.iter_mut().zip(eig_fft.iter()) {
      let lambda = eig.re / norm;
      *dst = if lambda > T::zero() {
        lambda.sqrt()
      } else {
        T::zero()
      };
    }

    let scale_n = out_len.max(1);

    Self {
      hurst,
      n,
      offset,
      out_len,
      t,
      scale: T::from_usize_(scale_n).powf(-hurst) * t.unwrap_or(T::one()).powf(hurst),
      sqrt_eigenvalues: Arc::new(sqrt_eigenvalues),
      fft_handler: Arc::new(FftHandler::new(2 * n)),
    }
  }

  pub(crate) fn sample_cpu(&self) -> Array1<T> {
    let len = 2 * self.n;
    let mut fgn = Array1::<T>::zeros(self.out_len);

    T::with_fgn_complex_scratch(len, |rnd| {
      // SAFETY: Complex<T> is repr(C) with layout {re: T, im: T}, identical to [T; 2]
      let flat = unsafe { std::slice::from_raw_parts_mut(rnd.as_mut_ptr() as *mut T, 2 * len) };
      T::fill_standard_normal_slice(flat);
      for (z, &w) in rnd.iter_mut().zip(self.sqrt_eigenvalues.iter()) {
        z.re = z.re * w;
        z.im = z.im * w;
      }

      let mut rnd_view = ArrayViewMut1::from(rnd);
      ndfft_inplace_par(&mut rnd_view, &*self.fft_handler, 0);
      let src = rnd_view.slice(s![1..self.out_len + 1]);
      for (dst, c) in fgn.iter_mut().zip(src.iter()) {
        *dst = c.re * self.scale;
      }
    });

    fgn
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;

  use super::FGN;
  use crate::stats::fd::FractalDim;

  fn generate_fgn_paths(h: f64, n: usize, t: f64, m: usize) -> Vec<Vec<f64>> {
    let fgn = FGN::<f64>::new(h, n, Some(t));
    let mut out = Vec::with_capacity(m);
    for _ in 0..m {
      out.push(fgn.sample_cpu().to_vec());
    }
    out
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

  fn nearest_quantile(sorted: &[f64], p: f64) -> f64 {
    let idx = (((sorted.len() - 1) as f64) * p).round() as usize;
    sorted[idx]
  }

  #[test]
  fn dt_and_scale_use_requested_length_not_fft_padding() {
    let hs = [0.2_f64, 0.7_f64];
    let ns = [3_usize, 17, 1000, 4095];
    let ts = [0.7_f64, 2.0_f64];

    for &h in &hs {
      for &n in &ns {
        for &t in &ts {
          let fgn = FGN::<f64>::new(h, n, Some(t));

          // Internal FFT length is padded, but dt/scale must follow requested n.
          assert!(fgn.n >= n && fgn.n.is_power_of_two());
          assert!((fgn.dt() - (t / n as f64)).abs() < 1e-15);

          let expected_scale = (n as f64).powf(-h) * t.powf(h);
          assert!((fgn.scale - expected_scale).abs() < 1e-15);
        }
      }
    }
  }

  #[test]
  fn fgn_marginal_distribution_and_covariance_match_theory() {
    let h = 0.72_f64;
    let n = 2048_usize;
    let t = 1.0_f64;
    let m = 1024_usize;
    let paths = generate_fgn_paths(h, n, t, m);

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
    let std = var.sqrt();

    let m3 = values
      .iter()
      .map(|x| {
        let d = *x - mean;
        d * d * d
      })
      .sum::<f64>()
      / count;
    let m4 = values
      .iter()
      .map(|x| {
        let d = *x - mean;
        d * d * d * d
      })
      .sum::<f64>()
      / count;
    let skew = m3 / std.powi(3);
    let excess_kurtosis = m4 / std.powi(4) - 3.0;

    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q025 = (nearest_quantile(&sorted, 0.025) - mean) / std;
    let q975 = (nearest_quantile(&sorted, 0.975) - mean) / std;

    let dt = t / n as f64;
    let var_theory = dt.powf(2.0 * h);
    let cov1_theory = var_theory * unit_lag_covariance(h, 1);
    let cov4_theory = var_theory * unit_lag_covariance(h, 4);

    let cov1_emp = lag_covariance(&paths, mean, 1);
    let cov4_emp = lag_covariance(&paths, mean, 4);

    assert!(mean.abs() < 5e-4, "mean too far from zero: {mean}");
    assert!(
      ((var / var_theory) - 1.0).abs() < 0.05,
      "variance mismatch: emp={var}, theory={var_theory}"
    );
    assert!(
      (skew.abs() < 0.05) && (excess_kurtosis.abs() < 0.10),
      "non-Gaussian marginals: skew={skew}, excess_kurtosis={excess_kurtosis}"
    );
    assert!(
      (q025 + 1.96).abs() < 0.10 && (q975 - 1.96).abs() < 0.10,
      "quantile mismatch: q025={q025}, q975={q975}"
    );
    assert!(
      ((cov1_emp / cov1_theory) - 1.0).abs() < 0.05,
      "lag-1 covariance mismatch: emp={cov1_emp}, theory={cov1_theory}"
    );
    assert!(
      ((cov4_emp / cov4_theory) - 1.0).abs() < 0.05,
      "lag-4 covariance mismatch: emp={cov4_emp}, theory={cov4_theory}"
    );
  }

  #[test]
  fn fgn_lag1_correlation_sign_matches_hurst_regime() {
    let n = 2048_usize;
    let t = 1.0_f64;
    let m = 192_usize;

    let low_h = 0.25_f64;
    let high_h = 0.80_f64;

    let low_paths = generate_fgn_paths(low_h, n, t, m);
    let high_paths = generate_fgn_paths(high_h, n, t, m);

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

    let low_cov1 = lag_covariance(&low_paths, low_mean, 1);
    let high_cov1 = lag_covariance(&high_paths, high_mean, 1);

    let low_rho1 = low_cov1 / low_var;
    let high_rho1 = high_cov1 / high_var;

    assert!(
      low_rho1 < -0.10,
      "expected negative lag-1 correlation, got {low_rho1}"
    );
    assert!(
      high_rho1 > 0.10,
      "expected positive lag-1 correlation, got {high_rho1}"
    );
  }

  #[test]
  fn fbm_hurst_and_fractal_dimension_from_stats_fd_module() {
    let h = 0.78_f64;
    let n = 4096_usize;
    let t = 1.0_f64;
    let m = 240_usize;
    let kmax = 32_usize;

    let fgn = FGN::<f64>::new(h, n, Some(t));
    let mut endpoints = Vec::with_capacity(m);
    let mut d_vario_sum = 0.0_f64;
    let mut d_higuchi_sum = 0.0_f64;
    let mut d_higuchi_count = 0usize;

    for path_idx in 0..m {
      let inc = fgn.sample_cpu();
      let mut fbm = vec![0.0_f64; n + 1];
      for i in 0..n {
        fbm[i + 1] = fbm[i] + inc[i];
      }
      endpoints.push(fbm[n]);

      let fd = FractalDim::new(Array1::from_vec(fbm));
      d_vario_sum += fd.variogram(Some(2.0));

      // Higuchi is computationally heavier, sample it on a subset.
      if path_idx % 2 == 0 {
        d_higuchi_sum += fd.higuchi_fd(kmax);
        d_higuchi_count += 1;
      }
    }

    let fractal_dim_vario = d_vario_sum / m as f64;
    let fractal_dim_higuchi = d_higuchi_sum / d_higuchi_count as f64;
    let h_from_vario = 2.0 - fractal_dim_vario;
    let fractal_dim_theory = 2.0 - h;

    let endpoint_mean = endpoints.iter().sum::<f64>() / endpoints.len() as f64;
    let endpoint_var = endpoints
      .iter()
      .map(|x| {
        let d = *x - endpoint_mean;
        d * d
      })
      .sum::<f64>()
      / endpoints.len() as f64;

    assert!(
      (h_from_vario - h).abs() < 0.05,
      "H mismatch from variogram FD: h_est={h_from_vario}, h={h}"
    );
    assert!(
      (fractal_dim_vario - fractal_dim_theory).abs() < 0.05,
      "variogram FD mismatch: D_est={fractal_dim_vario}, D={fractal_dim_theory}"
    );
    assert!(
      (fractal_dim_higuchi - fractal_dim_theory).abs() < 0.05,
      "Higuchi FD mismatch: D_est={fractal_dim_higuchi}, D={fractal_dim_theory}"
    );
    assert!(
      ((endpoint_var / (t.powf(2.0 * h))) - 1.0).abs() < 0.18,
      "endpoint variance mismatch: emp={endpoint_var}, theory={}",
      t.powf(2.0 * h)
    );
  }
}
