use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::cev::Cev;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::diffusion::ou::Ou;

use crate::mle::DensityApprox;
use crate::mle::fit_mle;
use crate::traits::ProcessExt;

#[test]
fn gbm_process_ext_with_mle() {
  let gbm = Gbm::new(
    0.05,
    0.2,
    2501,
    Some(100.0),
    Some(10.0),
    Deterministic::new(42),
  );
  let path = gbm.sample();
  assert_eq!(path.len(), 2501);

  let dt = 10.0 / 2500.0;
  let mut gbm_fit = Gbm::new(0.0, 0.5, 100, Some(100.0), Some(1.0), Deterministic::new(0));
  let result = fit_mle(&mut gbm_fit, path.view(), dt, DensityApprox::Euler, None);
  assert!(
    (result.params[1] - 0.2).abs() < 0.15,
    "sigma estimate too far: {} vs 0.2",
    result.params[1]
  );
}

#[test]
fn ou_process_ext_with_mle() {
  let ou = Ou::new(
    2.0,
    1.0,
    0.3,
    2501,
    Some(1.0),
    Some(10.0),
    Deterministic::new(123),
  );
  let path = ou.sample();
  assert_eq!(path.len(), 2501);

  let dt = 10.0 / 2500.0;
  let mut ou_fit = Ou::new(
    1.0,
    0.5,
    0.5,
    100,
    Some(1.0),
    Some(1.0),
    Deterministic::new(0),
  );
  let result = fit_mle(&mut ou_fit, path.view(), dt, DensityApprox::Exact, None);
  assert!(
    (result.params[1] - 1.0).abs() < 0.5,
    "mu estimate too far: {} vs 1.0",
    result.params[1]
  );
  assert!(
    (result.params[2] - 0.3).abs() < 0.15,
    "sigma estimate too far: {} vs 0.3",
    result.params[2]
  );
}

#[test]
fn cir_process_ext_with_mle() {
  let cir = Cir::new(
    2.0,
    0.04,
    0.1,
    5001,
    Some(0.04),
    Some(20.0),
    None,
    Deterministic::new(55),
  );
  let path = cir.sample();
  assert_eq!(path.len(), 5001);

  let dt = 20.0 / 5000.0;
  let mut cir_fit = Cir::new(
    1.0,
    0.05,
    0.2,
    100,
    Some(0.04),
    Some(1.0),
    None,
    Deterministic::new(0),
  );
  let result = fit_mle(&mut cir_fit, path.view(), dt, DensityApprox::Euler, None);
  assert!(
    (result.params[1] - 0.04).abs() < 0.03,
    "mu estimate too far: {} vs 0.04",
    result.params[1]
  );
}

#[test]
fn ou_sample_then_mle_roundtrip() {
  let ou = Ou::new(
    3.0,
    2.0,
    0.5,
    10001,
    Some(2.0),
    Some(10.0),
    Deterministic::new(77),
  );
  let path: Array1<f64> = ProcessExt::<f64>::sample(&ou);

  let dt = 10.0 / 10000.0;
  let mut ou_fit = Ou::new(
    1.0,
    1.0,
    1.0,
    100,
    Some(1.0),
    Some(1.0),
    Deterministic::new(0),
  );
  let result = fit_mle(&mut ou_fit, path.view(), dt, DensityApprox::Exact, None);

  assert!(
    (result.params[0] - 3.0).abs() < 2.0,
    "theta estimate: {} vs 3.0",
    result.params[0]
  );
  assert!(
    (result.params[1] - 2.0).abs() < 0.5,
    "mu estimate: {} vs 2.0",
    result.params[1]
  );
  assert!(
    (result.params[2] - 0.5).abs() < 0.2,
    "sigma estimate: {} vs 0.5",
    result.params[2]
  );
}

#[test]
fn cev_process_ext_with_mle() {
  let cev = Cev::new(
    0.05,
    0.3,
    0.7,
    2501,
    Some(100.0),
    Some(10.0),
    Deterministic::new(42),
  );
  let path = cev.sample();
  assert_eq!(path.len(), 2501);

  let dt = 10.0 / 2500.0;
  let mut cev_fit = Cev::new(
    0.0,
    0.5,
    0.5,
    100,
    Some(100.0),
    Some(1.0),
    Deterministic::new(0),
  );
  let result = fit_mle(&mut cev_fit, path.view(), dt, DensityApprox::Euler, None);

  assert!(
    (result.params[1] - 0.3).abs() < 0.2,
    "sigma estimate too far: {} vs 0.3",
    result.params[1]
  );
  assert!(
    (result.params[2] - 0.7).abs() < 0.5,
    "gamma estimate too far: {} vs 0.7",
    result.params[2]
  );
}
