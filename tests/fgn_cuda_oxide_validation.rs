//! Validates the experimental cuda-oxide FGN backend against covariance
//! structure. Requires building/running through `cargo oxide`.

#[cfg(feature = "cuda-oxide-experimental")]
mod cuda_oxide_validation {
  use either::Either;
  use stochastic_rs::stochastic::noise::fgn::Fgn;
  use stochastic_rs::traits::ProcessExt;

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

  fn cuda_oxide_paths(h: f32, n: usize, m: usize) -> Vec<Vec<f64>> {
    let fgn = Fgn::<f32>::new(h, n, Some(1.0));
    match fgn
      .sample_cuda_oxide_with_module(m, "fgn_cuda_oxide_validation")
      .expect("cuda-oxide sampling failed")
    {
      Either::Left(path) => vec![path.iter().map(|&x| x as f64).collect()],
      Either::Right(paths) => paths
        .outer_iter()
        .map(|row| row.iter().map(|&x| x as f64).collect())
        .collect(),
    }
  }

  #[test]
  fn cuda_oxide_fgn_variance_and_lag_sign() {
    let h = 0.72_f32;
    let n = 512_usize;
    let m = 2048_usize;
    let paths = cuda_oxide_paths(h, n, m);

    let values: Vec<f64> = paths.iter().flatten().copied().collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let var_theory = (1.0 / n as f64).powf(2.0 * h as f64);
    let cov1 = lag_covariance(&paths, mean, 1);

    assert!(mean.abs() < 0.01, "mean too far from zero: {mean}");
    assert!(
      ((var / var_theory) - 1.0).abs() < 0.15,
      "variance ratio: {}",
      var / var_theory
    );
    assert!(
      cov1 > 0.0,
      "H>0.5 should have positive lag-1 covariance: {cov1}"
    );
  }

  #[test]
  fn cuda_oxide_non_power_of_two_n_shape() {
    let fgn = Fgn::<f32>::new(0.7, 3000, Some(1.0));
    let out = fgn
      .sample_cuda_oxide_with_module(8, "fgn_cuda_oxide_validation")
      .expect("cuda-oxide batch");
    let batch = out.right().expect("m>1 returns Array2");
    assert_eq!(batch.shape(), &[8, 3000]);
  }
}
