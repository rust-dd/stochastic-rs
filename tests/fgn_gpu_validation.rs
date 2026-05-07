//! Validates the CubeCL GPU FFT implementation by comparing the empirical
//! covariance structure of GPU-sampled fGN against theoretical values.
//!
//! The autocovariance of fGN increments is:
//! $$\gamma(k) = \tfrac12\bigl(|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H}\bigr)$$
//!
//! If the GPU FFT is correct, empirical covariance from GPU samples must
//! match this formula to within Monte Carlo noise.

#[cfg(feature = "gpu-wgpu")]
mod gpu_fft_validation {
  use either::Either;
  use stochastic_rs::stochastic::noise::fgn::Fgn;
  use stochastic_rs::traits::ProcessExt;

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

  fn sample_gpu_paths(h: f32, n: usize, t: f32, m: usize) -> Vec<Vec<f64>> {
    let fgn = Fgn::<f32>::new(h, n, Some(t));
    match fgn.sample_gpu(m).expect("GPU sampling failed") {
      Either::Left(path) => vec![path.iter().map(|&x| x as f64).collect()],
      Either::Right(paths) => paths
        .outer_iter()
        .map(|row| row.iter().map(|&x| x as f64).collect())
        .collect(),
    }
  }

  fn sample_cpu_paths(h: f64, n: usize, t: f64, m: usize) -> Vec<Vec<f64>> {
    let fgn = Fgn::<f64>::new(h, n, Some(t));
    fgn
      .sample_par(m)
      .into_iter()
      .map(|path| path.to_vec())
      .collect()
  }

  #[test]
  fn gpu_fgn_covariance_matches_theory() {
    let h = 0.72_f32;
    let n = 512_usize;
    let t = 1.0_f32;
    let m = 2048_usize;

    let paths = sample_gpu_paths(h, n, t, m);

    let mut values = Vec::with_capacity(m * n);
    for p in &paths {
      values.extend_from_slice(p);
    }

    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count;

    let h64 = h as f64;
    let dt = (t as f64) / n as f64;
    let var_theory = dt.powf(2.0 * h64);
    let cov1_theory = var_theory * unit_lag_covariance(h64, 1);
    let cov4_theory = var_theory * unit_lag_covariance(h64, 4);
    let cov1_emp = lag_covariance(&paths, mean, 1);
    let cov4_emp = lag_covariance(&paths, mean, 4);

    eprintln!("GPU fGN validation (H={h}, n={n}, m={m}):");
    eprintln!("  mean:     {mean:.6} (expect ~0)");
    eprintln!(
      "  variance: {var:.6} (theory {var_theory:.6}, ratio {:.4})",
      var / var_theory
    );
    eprintln!(
      "  cov(1):   {cov1_emp:.6} (theory {cov1_theory:.6}, ratio {:.4})",
      cov1_emp / cov1_theory
    );
    eprintln!(
      "  cov(4):   {cov4_emp:.6} (theory {cov4_theory:.6}, ratio {:.4})",
      cov4_emp / cov4_theory
    );

    assert!(mean.abs() < 0.01, "mean too far from zero: {mean}");
    assert!(
      ((var / var_theory) - 1.0).abs() < 0.15,
      "variance mismatch: emp={var}, theory={var_theory}"
    );
    assert!(
      ((cov1_emp / cov1_theory) - 1.0).abs() < 0.20,
      "lag-1 covariance mismatch: emp={cov1_emp}, theory={cov1_theory}"
    );
    assert!(
      ((cov4_emp / cov4_theory) - 1.0).abs() < 0.25,
      "lag-4 covariance mismatch: emp={cov4_emp}, theory={cov4_theory}"
    );
  }

  #[test]
  fn gpu_fgn_lag1_sign_matches_hurst() {
    let n = 512_usize;
    let t = 1.0_f32;
    let m = 4096_usize;

    let low_paths = sample_gpu_paths(0.25, n, t, m);
    let high_paths = sample_gpu_paths(0.80, n, t, m);

    let low_vals: Vec<f64> = low_paths.iter().flatten().copied().collect();
    let high_vals: Vec<f64> = high_paths.iter().flatten().copied().collect();
    let low_mean = low_vals.iter().sum::<f64>() / low_vals.len() as f64;
    let high_mean = high_vals.iter().sum::<f64>() / high_vals.len() as f64;
    let low_var =
      low_vals.iter().map(|x| (x - low_mean).powi(2)).sum::<f64>() / low_vals.len() as f64;
    let high_var = high_vals
      .iter()
      .map(|x| (x - high_mean).powi(2))
      .sum::<f64>()
      / high_vals.len() as f64;

    let low_rho1 = lag_covariance(&low_paths, low_mean, 1) / low_var;
    let high_rho1 = lag_covariance(&high_paths, high_mean, 1) / high_var;

    eprintln!("GPU lag-1 correlation: H=0.25 -> rho1={low_rho1:.4}, H=0.80 -> rho1={high_rho1:.4}");

    assert!(
      low_rho1 < -0.05,
      "expected negative lag-1 for H<0.5, got {low_rho1}"
    );
    assert!(
      high_rho1 > 0.05,
      "expected positive lag-1 for H>0.5, got {high_rho1}"
    );
  }

  #[test]
  fn gpu_vs_cpu_covariance_structure_matches() {
    let h = 0.72;
    let n = 512_usize;
    let t = 1.0;
    let m = 2048_usize;

    let cpu_paths = sample_cpu_paths(h, n, t, m);
    let gpu_paths = sample_gpu_paths(h as f32, n, t as f32, m);

    let cpu_vals: Vec<f64> = cpu_paths.iter().flatten().copied().collect();
    let gpu_vals: Vec<f64> = gpu_paths.iter().flatten().copied().collect();

    let cpu_mean = cpu_vals.iter().sum::<f64>() / cpu_vals.len() as f64;
    let gpu_mean = gpu_vals.iter().sum::<f64>() / gpu_vals.len() as f64;
    let cpu_var =
      cpu_vals.iter().map(|x| (x - cpu_mean).powi(2)).sum::<f64>() / cpu_vals.len() as f64;
    let gpu_var =
      gpu_vals.iter().map(|x| (x - gpu_mean).powi(2)).sum::<f64>() / gpu_vals.len() as f64;

    let cpu_cov1 = lag_covariance(&cpu_paths, cpu_mean, 1);
    let gpu_cov1 = lag_covariance(&gpu_paths, gpu_mean, 1);
    let cpu_cov4 = lag_covariance(&cpu_paths, cpu_mean, 4);
    let gpu_cov4 = lag_covariance(&gpu_paths, gpu_mean, 4);

    eprintln!("CPU vs GPU covariance comparison (H={h}, n={n}, m={m}):");
    eprintln!(
      "  variance: CPU={cpu_var:.6}, GPU={gpu_var:.6}, ratio={:.4}",
      gpu_var / cpu_var
    );
    eprintln!(
      "  cov(1):   CPU={cpu_cov1:.6}, GPU={gpu_cov1:.6}, ratio={:.4}",
      gpu_cov1 / cpu_cov1
    );
    eprintln!(
      "  cov(4):   CPU={cpu_cov4:.6}, GPU={gpu_cov4:.6}, ratio={:.4}",
      gpu_cov4 / cpu_cov4
    );

    assert!(
      ((gpu_var / cpu_var) - 1.0).abs() < 0.15,
      "variance divergence: CPU={cpu_var}, GPU={gpu_var}"
    );
    assert!(
      ((gpu_cov1 / cpu_cov1) - 1.0).abs() < 0.20,
      "lag-1 cov divergence: CPU={cpu_cov1}, GPU={gpu_cov1}"
    );
    assert!(
      ((gpu_cov4 / cpu_cov4) - 1.0).abs() < 0.25,
      "lag-4 cov divergence: CPU={cpu_cov4}, GPU={gpu_cov4}"
    );
  }
}
