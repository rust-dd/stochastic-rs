use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::volterra::kernel::ExponentialKernel;
use crate::volterra::sve::VolterraSde;

const N: usize = 512;

fn kernel() -> ExponentialKernel<f64> {
  ExponentialKernel::new(1.5, 1.0)
}

fn zero_drift(_t: f64, _x: f64) -> f64 {
  0.0
}

fn unit_diffusion(_t: f64, _x: f64) -> f64 {
  1.0
}

/// The type's defining identity: its output is exactly $p$ applied pointwise
/// to the Gaussian Volterra path the same kernel and seed produce through
/// [`VolterraSde`]. Asserted to machine precision rather than statistically,
/// because it is an algebraic identity, not an approximation — the two share
/// the same lift, the same kernel and the same derived seed.
#[test]
fn output_is_the_polynomial_of_the_driving_gaussian_volterra_path() {
  let coefficients = array![0.05, 0.3, -0.1, 0.02];
  let gpv = GaussianPolynomialVolatility::new(
    kernel(),
    coefficients.clone(),
    N,
    Some(1.0),
    Deterministic::new(11),
  );
  let driver = VolterraSde::new(
    kernel(),
    zero_drift as fn(f64, f64) -> f64,
    unit_diffusion as fn(f64, f64) -> f64,
    N,
    Some(0.0),
    Some(1.0),
    Deterministic::new(11),
  );

  let sigma = gpv.sample();
  let x = driver.sample();
  assert_eq!(sigma.len(), x.len());

  for (i, (&s, &xi)) in sigma.iter().zip(x.iter()).enumerate() {
    let expected = gpv.evaluate_polynomial(xi);
    assert!(
      (s - expected).abs() <= 1e-14 * expected.abs().max(1.0),
      "step {i}: sigma={s} but p(x)={expected} for x={xi}"
    );
  }
}

/// Horner must agree with the direct power sum on values where both are
/// well conditioned. This guards the evaluation order, not the model.
#[test]
fn horner_matches_the_direct_power_sum() {
  let coefficients = array![0.05, 0.3, -0.1, 0.02, 0.004, -0.0007];
  let gpv = GaussianPolynomialVolatility::quintic(
    kernel(),
    coefficients.clone(),
    N,
    Some(1.0),
    Deterministic::new(3),
  );

  for &x in &[-2.0_f64, -0.5, 0.0, 0.25, 1.0, 3.0] {
    let direct: f64 = coefficients
      .iter()
      .enumerate()
      .map(|(k, c)| c * x.powi(k as i32))
      .sum();
    let horner = gpv.evaluate_polynomial(x);
    assert!(
      (horner - direct).abs() <= 1e-13 * direct.abs().max(1.0),
      "x={x}: horner={horner} direct={direct}"
    );
  }
}

/// A constant polynomial must give a constant path — the cheapest possible
/// check that the coefficients are actually reaching the output rather than
/// being ignored.
#[test]
fn a_constant_polynomial_gives_a_constant_path() {
  let gpv =
    GaussianPolynomialVolatility::new(kernel(), array![0.2], N, Some(1.0), Deterministic::new(5));
  for &v in gpv.sample().iter() {
    assert_eq!(v, 0.2);
  }
}

#[test]
fn is_seed_reproducible() {
  let coefficients = array![0.05, 0.3, -0.1];
  let a = GaussianPolynomialVolatility::new(
    kernel(),
    coefficients.clone(),
    N,
    Some(1.0),
    Deterministic::new(9),
  )
  .sample();
  let b =
    GaussianPolynomialVolatility::new(kernel(), coefficients, N, Some(1.0), Deterministic::new(9))
      .sample();
  assert_eq!(a, b);
}

#[test]
#[should_panic(expected = "coefficients must contain at least a constant term")]
fn rejects_an_empty_polynomial() {
  let _ = GaussianPolynomialVolatility::new(
    kernel(),
    Array1::<f64>::zeros(0),
    N,
    None,
    Deterministic::new(1),
  );
}

#[test]
#[should_panic(expected = "a quintic polynomial needs exactly six coefficients")]
fn quintic_rejects_the_wrong_degree() {
  let _ = GaussianPolynomialVolatility::quintic(
    kernel(),
    array![1.0, 2.0, 3.0],
    N,
    None,
    Deterministic::new(1),
  );
}

#[test]
#[should_panic(expected = "n must be at least 2")]
fn rejects_n_below_two() {
  let _ = GaussianPolynomialVolatility::new(kernel(), array![0.2], 1, None, Deterministic::new(1));
}
