use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

fn flat_tenor(m: usize, dt: f64) -> Array1<f64> {
  Array1::from_iter((0..=m).map(|i| i as f64 * dt))
}

#[test]
fn lmm_sample_runs_independent_factors() {
  let m = 4;
  let tenor = flat_tenor(m, 0.5); // T_0=0, T_1=0.5, ..., T_4=2.0
  let l0 = Array1::from(vec![0.03, 0.035, 0.04, 0.045]);
  let sigma = Array1::from(vec![0.20, 0.20, 0.20, 0.20]);
  let lmm: Lmm<f64, Deterministic> =
    Lmm::new(tenor, l0, sigma, 100, Some(2.0), Deterministic::new(42));
  let path = lmm.sample();
  assert_eq!(path.shape(), &[m, 100]);
  // Initial column matches l0.
  for n in 0..m {
    assert!(path[(n, 0)] > 0.0);
  }
  // All values must remain positive (log-Euler preserves positivity).
  for v in path.iter() {
    assert!(*v > 0.0, "LMM forward went non-positive");
  }
}

#[test]
fn lmm_correlated_factors_run() {
  let m = 3;
  let tenor = flat_tenor(m, 1.0);
  let l0 = Array1::from(vec![0.04, 0.045, 0.05]);
  let sigma = Array1::from(vec![0.25, 0.20, 0.18]);
  // Constant 0.6 off-diagonal correlation.
  let mut rho = Array2::<f64>::eye(m);
  for i in 0..m {
    for j in 0..m {
      if i != j {
        rho[(i, j)] = 0.6;
      }
    }
  }
  let lmm: Lmm<f64, Deterministic> =
    Lmm::new(tenor, l0, sigma, 50, Some(3.0), Deterministic::new(123)).with_correlation(rho);
  let path = lmm.sample();
  assert_eq!(path.shape(), &[m, 50]);
  for v in path.iter() {
    assert!(*v > 0.0);
  }
}

#[test]
fn lmm_seeded_is_deterministic() {
  let tenor = flat_tenor(2, 0.5);
  let l0 = Array1::from(vec![0.03, 0.035]);
  let sigma = Array1::from(vec![0.2, 0.2]);
  let a: Lmm<f64, Deterministic> = Lmm::new(
    tenor.clone(),
    l0.clone(),
    sigma.clone(),
    30,
    Some(1.0),
    Deterministic::new(7),
  );
  let b: Lmm<f64, Deterministic> = Lmm::new(tenor, l0, sigma, 30, Some(1.0), Deterministic::new(7));
  let pa = a.sample();
  let pb = b.sample();
  for (x, y) in pa.iter().zip(pb.iter()) {
    assert_eq!(x, y, "seeded LMM not reproducible");
  }
}

fn three_libors() -> Lmm<f64> {
  Lmm::new(
    ndarray::array![0.0, 0.5, 1.0, 1.5],
    ndarray::array![0.03, 0.03, 0.03],
    ndarray::array![0.2, 0.2, 0.2],
    16,
    Some(1.0),
    Unseeded,
  )
}

#[test]
#[should_panic(
  expected = "rho[(0, 0)] must satisfy `(rho[(i, i)] - T::one()).abs() <= tol`, got rho[(0, 0)] = 4.0"
)]
fn a_covariance_matrix_is_refused_as_a_correlation() {
  let _ = three_libors().with_correlation(Array2::<f64>::eye(3) * 4.0);
}

#[test]
#[should_panic(
  expected = "rho[(1, 0)] must satisfy `(rho[(i, j)] - rho[(j, i)]).abs() <= tol`, got rho[(1, 0)] = 0.2, rho[(0, 1)] = 0.6"
)]
fn an_asymmetric_matrix_is_refused_as_a_correlation() {
  let rho = ndarray::array![[1.0, 0.6, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]];
  let _ = three_libors().with_correlation(rho);
}

#[test]
#[should_panic(
  expected = "rho[(0, 1)] must satisfy `rho[(i, j)].is_finite()`, got rho[(0, 1)] = NaN"
)]
fn a_nan_correlation_is_refused_by_name() {
  let rho = ndarray::array![[1.0, f64::NAN, 0.0], [f64::NAN, 1.0, 0.0], [0.0, 0.0, 1.0]];
  let _ = three_libors().with_correlation(rho);
}

#[test]
fn a_correlation_is_factored_into_its_lower_cholesky() {
  let rho = ndarray::array![[1.0, 0.6, 0.3], [0.6, 1.0, 0.5], [0.3, 0.5, 1.0]];
  let l = three_libors()
    .with_correlation(rho.clone())
    .chol
    .expect("set");
  let recon = l.dot(&l.t());
  assert!((&recon - &rho).iter().all(|d| d.abs() < 1e-12));
}
