//! Integration tests for the `quant::factors` module.
#![cfg(feature = "openblas")]

use ndarray::Array2;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::quant::factors::PairsSignal;
use stochastic_rs::quant::factors::fama_macbeth;
use stochastic_rs::quant::factors::ledoit_wolf_shrinkage;
use stochastic_rs::quant::factors::pairs_signals;
use stochastic_rs::quant::factors::pca_decompose;
use stochastic_rs::quant::factors::sample_covariance;

fn standard_normal_matrix(seed: u64, t: usize, p: usize) -> Array2<f64> {
  let dist = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(seed));
  let mut buf = vec![0.0_f64; t * p];
  dist.fill_slice_fast(&mut buf);
  Array2::from_shape_vec((t, p), buf).unwrap()
}

#[test]
fn shrinkage_matrix_within_unit_interval() {
  let r = standard_normal_matrix(11, 100, 8);
  let lw = ledoit_wolf_shrinkage(r.view());
  assert!((0.0..=1.0).contains(&lw.alpha));
  let s = sample_covariance(r.view());
  for i in 0..8 {
    for j in 0..8 {
      let mix = lw.alpha * if i == j { lw.mu } else { 0.0 } + (1.0 - lw.alpha) * s[[i, j]];
      assert!((lw.covariance[[i, j]] - mix).abs() < 1e-12);
    }
  }
}

#[test]
fn pca_collinear_factor_dominates_variance() {
  let dist = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(23));
  let mut z = vec![0.0_f64; 500];
  dist.fill_slice_fast(&mut z);
  let mut x = Array2::<f64>::zeros((500, 5));
  for t in 0..500 {
    x[[t, 0]] = z[t];
    x[[t, 1]] = -1.5 * z[t];
    x[[t, 2]] = 0.7 * z[t];
    x[[t, 3]] = 2.1 * z[t];
    x[[t, 4]] = 1.3 * z[t];
  }
  let res = pca_decompose(x.view(), 0);
  assert!(res.explained_variance_ratio[0] > 0.999);
}

#[test]
fn fama_macbeth_finite_under_no_factor_signal() {
  let returns = standard_normal_matrix(31, 200, 10);
  let factors = standard_normal_matrix(33, 200, 1);
  let res = fama_macbeth(returns.view(), factors.view());
  assert!(res.gamma.iter().all(|v| v.is_finite()));
  assert!(res.std_errors.iter().all(|&v| v.is_finite()));
}

#[test]
fn pairs_round_trip_position_management() {
  use ndarray::Array1;
  let n = 600usize;
  let dist = SimdNormal::<f64>::new(0.0, 0.005, &Deterministic::new(41));
  let mut shocks = vec![0.0_f64; n];
  dist.fill_slice_fast(&mut shocks);
  let mut x = Array1::<f64>::zeros(n);
  let mut y = Array1::<f64>::zeros(n);
  for i in 0..n {
    x[i] = 100.0 + (i as f64) * 0.05;
    y[i] = 1.5 * x[i] + 0.5 + shocks[i];
  }
  y[300] += 0.5;
  let res = pairs_signals(y.view(), x.view(), 1.5, 0.5);
  let opens = res
    .signals
    .iter()
    .filter(|&&s| !matches!(s, PairsSignal::Flat))
    .count();
  assert!(opens >= 1);
  assert!(res.beta > 1.0);
}
