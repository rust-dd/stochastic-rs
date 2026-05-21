use ndarray::Array1;
use ndarray::Array2;
use ndarray::array;
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::NoiseModel;
use super::Sde;
use super::SdeMethod;

/// Geometric Brownian motion `dS = μS dt + σS dW` has analytical mean
/// `E[S_T] = S_0 · exp(μ·T)`. Each method should hit it within MC tolerance.
fn gbm_sde()
-> Sde<f64, impl Fn(&Array1<f64>, f64) -> Array1<f64>, impl Fn(&Array1<f64>, f64) -> Array2<f64>> {
  let mu = 0.05;
  let sigma = 0.20;
  Sde::new(
    move |x: &Array1<f64>, _t: f64| array![mu * x[0]],
    move |x: &Array1<f64>, _t: f64| Array2::from_elem((1, 1), sigma * x[0]),
    NoiseModel::Gaussian,
    None,
  )
}

fn final_mean(paths: &ndarray::Array3<f64>) -> f64 {
  let n_paths = paths.shape()[0];
  let last = paths.shape()[1] - 1;
  let sum: f64 = (0..n_paths).map(|p| paths[[p, last, 0]]).sum();
  sum / n_paths as f64
}

#[test]
fn euler_gbm_recovers_analytical_mean() {
  let sde = gbm_sde();
  let s0 = array![100.0_f64];
  let t = 1.0_f64;
  let dt = 1e-3_f64;
  let mut rng = StdRng::seed_from_u64(0x5DE_F00D);
  let paths = sde.solve(&s0, 0.0, t, dt, 4_000, SdeMethod::Euler, &mut rng);

  let analytic = 100.0_f64 * (0.05_f64 * t).exp();
  let m = final_mean(&paths);
  let rel_err = (m - analytic).abs() / analytic;
  assert!(rel_err < 0.02, "Euler relative error {rel_err}");
}

#[test]
fn milstein_gbm_recovers_analytical_mean() {
  let sde = gbm_sde();
  let s0 = array![100.0_f64];
  let t = 1.0_f64;
  let dt = 1e-3_f64;
  let mut rng = StdRng::seed_from_u64(0xC0FFEE);
  let paths = sde.solve(&s0, 0.0, t, dt, 4_000, SdeMethod::Milstein, &mut rng);

  let analytic = 100.0_f64 * (0.05_f64 * t).exp();
  let m = final_mean(&paths);
  let rel_err = (m - analytic).abs() / analytic;
  assert!(rel_err < 0.02, "Milstein relative error {rel_err}");
}

#[test]
fn srk2_and_srk4_produce_finite_paths() {
  let sde = gbm_sde();
  let s0 = array![100.0_f64];
  let mut rng = StdRng::seed_from_u64(0xBADCAFE);
  for method in [SdeMethod::SRK2, SdeMethod::SRK4] {
    let paths = sde.solve(&s0, 0.0, 0.5, 1e-3, 200, method, &mut rng);
    assert!(paths.iter().all(|v| v.is_finite()));
    for p in 0..paths.shape()[0] {
      assert!((paths[[p, 0, 0]] - 100.0).abs() < 1e-12);
    }
  }
}

/// Pure drift (σ ≡ 0) collapses the SDE to an ODE; Euler should track
/// `dx/dt = -k·x` exactly to first order and bracket the analytical
/// `e^{-k·T}` decay.
#[test]
fn pure_drift_decay_tracks_exponential() {
  let k = 1.0;
  let sde = Sde::new(
    move |x: &Array1<f64>, _t: f64| array![-k * x[0]],
    |_x: &Array1<f64>, _t: f64| Array2::<f64>::zeros((1, 1)),
    NoiseModel::Gaussian,
    None,
  );
  let s0 = array![1.0_f64];
  let t = 1.0_f64;
  let dt = 1e-3_f64;
  let mut rng = StdRng::seed_from_u64(0xDEC0DE);
  let paths = sde.solve(&s0, 0.0, t, dt, 1, SdeMethod::Euler, &mut rng);

  let last = paths.shape()[1] - 1;
  let computed = paths[[0, last, 0]];
  let analytic = (-k * t).exp();
  assert!(
    (computed - analytic).abs() < 5e-3,
    "Euler decay {computed} vs analytic {analytic}"
  );
}

/// Initial condition copied to row 0 across all paths and methods.
#[test]
fn initial_condition_preserved() {
  let sde = gbm_sde();
  let s0 = array![100.0_f64];
  let mut rng = StdRng::seed_from_u64(7);
  let paths = sde.solve(&s0, 0.0, 0.1, 1e-3, 5, SdeMethod::Euler, &mut rng);
  for p in 0..5 {
    assert_eq!(paths[[p, 0, 0]], 100.0);
  }
}
