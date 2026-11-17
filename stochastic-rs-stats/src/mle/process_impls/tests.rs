use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::cev::Cev;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::diffusion::ou::Ou;
use stochastic_rs_stochastic::params::VarianceBoundary;

use crate::mle::DensityApprox;
use crate::mle::DiffusionModel;
use crate::mle::fit_mle;
use crate::traits::ProcessExt;

#[test]
fn gbm_process_ext_with_mle() {
  let gbm = Gbm::new(0.05, 0.2, 2501, 100.0, Some(10.0), Deterministic::new(42));
  let path = gbm.sample();
  assert_eq!(path.len(), 2501);

  let dt = 10.0 / 2500.0;
  let mut gbm_fit = Gbm::new(0.0, 0.5, 100, 100.0, Some(1.0), Deterministic::new(0));
  let result = fit_mle(&mut gbm_fit, path.view(), dt, DensityApprox::Euler, None).unwrap();
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
  let result = fit_mle(&mut ou_fit, path.view(), dt, DensityApprox::Exact, None).unwrap();
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
    VarianceBoundary::Floor,
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
    VarianceBoundary::Floor,
    Deterministic::new(0),
  );
  let result = fit_mle(&mut cir_fit, path.view(), dt, DensityApprox::Euler, None).unwrap();
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
  let result = fit_mle(&mut ou_fit, path.view(), dt, DensityApprox::Exact, None).unwrap();

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
  let result = fit_mle(&mut cev_fit, path.view(), dt, DensityApprox::Euler, None).unwrap();

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

/// A daily step (`z ≈ 4025`), far past a truncated Bessel-I series; the reference is
/// `mpmath.besseli` at 80 digits.
#[test]
fn cir_exact_density_matches_mpmath_at_a_daily_step() {
  let cir = Cir::new(
    2.0,
    0.04,
    0.1,
    100,
    Some(0.04),
    Some(1.0),
    VarianceBoundary::Floor,
    Deterministic::new(0),
  );
  let density = DensityApprox::Exact.density(&cir, 0.04, 0.0405, 0.0, 0.004);
  let want = 290.067_400_877_071_3;
  assert!(
    ((density - want) / want).abs() < 1e-10,
    "Cir Exact: {density}"
  );
}

#[test]
fn cir_exact_mle_recovers_daily_parameters() {
  let cir = Cir::new(
    2.0,
    0.04,
    0.1,
    5001,
    Some(0.04),
    Some(20.0),
    VarianceBoundary::Floor,
    Deterministic::new(55),
  );
  let path = cir.sample();
  let dt = 20.0 / 5000.0;
  let mut cir_fit = Cir::new(
    1.0,
    0.05,
    0.2,
    100,
    Some(0.04),
    Some(1.0),
    VarianceBoundary::Floor,
    Deterministic::new(0),
  );
  let result = fit_mle(&mut cir_fit, path.view(), dt, DensityApprox::Exact, None).unwrap();
  assert!(
    (result.params[1] - 0.04).abs() < 0.01,
    "mu estimate: {} vs 0.04",
    result.params[1]
  );
  assert!(
    (result.params[2] - 0.1).abs() < 0.01,
    "sigma estimate: {} vs 0.1",
    result.params[2]
  );
}

/// A transition into a negative CIR state has no density: the exact law answers NaN.
#[test]
fn the_exact_cir_density_is_the_law() {
  let cir = Cir::new(
    2.0,
    0.04,
    0.3,
    2,
    None,
    None,
    VarianceBoundary::Floor,
    Deterministic::new(0),
  );
  let dt = 1.0 / 252.0;
  assert!(cir.exact_density(0.05, -0.01, 0.0, dt).unwrap().is_nan());
  for (x0, xt) in [(0.05, 0.051), (0.0, 0.01), (0.05, 0.0), (1e-40, 0.01)] {
    assert_eq!(
      cir.exact_density(x0, xt, 0.0, dt).unwrap().to_bits(),
      cir.transition_pdf(dt, x0, xt).to_bits(),
      "{x0} to {xt}"
    );
  }
}

/// A density is `0.0` outside its method's support and NaN where the method has none, as at a
/// variance of zero: no positive stand-in, which the objective would read as a likelihood.
#[test]
fn a_density_is_zero_outside_its_support_and_nan_where_it_has_none() {
  let dt = 1.0 / 252.0;
  let gbm = Gbm::new(0.05, 0.2, 100, 100.0, Some(1.0), Deterministic::new(0));
  assert_eq!(gbm.exact_density(100.0, 0.0, 0.0, dt), Some(0.0));
  assert_eq!(gbm.exact_density(100.0, -1.0, 0.0, dt), Some(0.0));
  assert!(gbm.exact_density(0.0, 100.0, 0.0, dt).unwrap().is_nan());
  assert!(gbm.exact_density(-1.0, -1.0, 0.0, dt).unwrap().is_nan());

  let mut ou = Ou::new(
    1.0,
    0.5,
    0.3,
    100,
    Some(0.5),
    Some(1.0),
    Deterministic::new(0),
  );
  // At this speed `1 − e^{−2θΔ}` rounds to zero.
  ou.set_params(&[1e-300, 0.5, 0.3]);
  assert!(ou.exact_density(0.5, 0.55, 0.0, dt).unwrap().is_nan());
  // Without noise Euler's variance is zero, and Kessler's at the level, where the drift is too.
  ou.set_params(&[1.0, 0.5, 0.0]);
  assert!(
    DensityApprox::Euler
      .density(&ou, 0.5, 0.55, 0.0, dt)
      .is_nan()
  );
  assert!(
    DensityApprox::Kessler
      .density(&ou, 0.5, 0.5, 0.0, dt)
      .is_nan()
  );

  let cir = Cir::new(
    2.0,
    0.04,
    0.3,
    100,
    Some(0.05),
    Some(1.0),
    VarianceBoundary::Floor,
    Deterministic::new(0),
  );
  // The Milstein step from 0.05 lands above `−1.7e-4`, and its density underflows far above it.
  assert_eq!(
    DensityApprox::Elerian.density(&cir, 0.05, -0.01, 0.0, dt),
    0.0
  );
  assert_eq!(
    DensityApprox::Elerian.density(&cir, 0.05, 90.0, 0.0, 2.2),
    0.0
  );
}
