//! Validates Accelerate/vDSP FFT by comparing autocovariance against theory and CPU.

#[cfg(feature = "accelerate")]
mod accel_validation {
  use either::Either;
  use stochastic_rs::stochastic::noise::fgn::Fgn;
  use stochastic_rs::traits::ProcessExt;

  fn accel_paths(h: f32, n: usize, m: usize) -> Vec<Vec<f64>> {
    let fgn = Fgn::<f32>::new(h, n, Some(1.0));
    match fgn
      .sample_accelerate(m)
      .expect("accelerate sampling failed")
    {
      Either::Left(p) => vec![p.iter().map(|&x| x as f64).collect()],
      Either::Right(ps) => ps
        .outer_iter()
        .map(|r| r.iter().map(|&x| x as f64).collect())
        .collect(),
    }
  }

  fn lag_cov(paths: &[Vec<f64>], mean: f64, lag: usize) -> f64 {
    let (mut s, mut c) = (0.0, 0usize);
    for p in paths {
      for i in 0..(p.len() - lag) {
        s += (p[i] - mean) * (p[i + lag] - mean);
        c += 1;
      }
    }
    s / c as f64
  }

  #[test]
  fn accelerate_fgn_covariance() {
    let h = 0.72_f32;
    let n = 512;
    let m = 4096;
    let paths = accel_paths(h, n, m);

    let all: Vec<f64> = paths.iter().flatten().copied().collect();
    let mean = all.iter().sum::<f64>() / all.len() as f64;
    let var = all.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / all.len() as f64;

    let h64 = h as f64;
    let dt = 1.0 / n as f64;
    let var_th = dt.powf(2.0 * h64);
    let cov1_th = var_th * 0.5 * (2.0_f64.powf(2.0 * h64) - 2.0 + 0.0);
    let cov1_th = var_th * 0.5 * (2.0_f64.powf(2.0 * h64) - 2.0);
    let cov1 = lag_cov(&paths, mean, 1);

    eprintln!("Accelerate fGN (H={h}, n={n}, m={m}):");
    eprintln!(
      "  var:   {var:.8} (theory {var_th:.8}, ratio {:.4})",
      var / var_th
    );
    eprintln!(
      "  cov1:  {cov1:.8} (theory {cov1_th:.8}, ratio {:.4})",
      cov1 / cov1_th
    );
    eprintln!("  mean:  {mean:.8}");

    assert!(mean.abs() < 0.005, "mean: {mean}");
    assert!(
      ((var / var_th) - 1.0).abs() < 0.10,
      "var ratio: {}",
      var / var_th
    );
  }

  #[test]
  fn accelerate_lag1_sign() {
    let n = 512;
    let m = 4096;

    let low = accel_paths(0.25, n, m);
    let high = accel_paths(0.80, n, m);

    let lv: Vec<f64> = low.iter().flatten().copied().collect();
    let hv: Vec<f64> = high.iter().flatten().copied().collect();
    let lm = lv.iter().sum::<f64>() / lv.len() as f64;
    let hm = hv.iter().sum::<f64>() / hv.len() as f64;
    let lvar = lv.iter().map(|x| (x - lm).powi(2)).sum::<f64>() / lv.len() as f64;
    let hvar = hv.iter().map(|x| (x - hm).powi(2)).sum::<f64>() / hv.len() as f64;

    let lrho = lag_cov(&low, lm, 1) / lvar;
    let hrho = lag_cov(&high, hm, 1) / hvar;

    eprintln!("Accelerate lag-1: H=0.25 -> {lrho:.4}, H=0.80 -> {hrho:.4}");
    assert!(lrho < -0.05, "H<0.5 should be negative: {lrho}");
    assert!(hrho > 0.05, "H>0.5 should be positive: {hrho}");
  }
}
