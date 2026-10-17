use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

fn unfitted(delta: f64) -> Bb7 {
  let mut c = Bb7::default();
  c.set_delta(delta);
  c
}

#[test]
fn h_function_is_the_v_derivative_of_the_cdf() {
  let c = Bb7::new(1.8, 0.9);
  for (u, v) in [(0.3, 0.6), (0.8, 0.2), (0.5, 0.5)] {
    let h = c.partial_derivative(array![[u, v]].view()).unwrap()[0];
    let eps = 1e-6;
    let up = c.cdf(array![[u, v + eps]].view()).unwrap()[0];
    let dn = c.cdf(array![[u, v - eps]].view()).unwrap()[0];
    assert!(
      (h - (up - dn) / (2.0 * eps)).abs() < 1e-6,
      "h {h} vs fd {}",
      (up - dn) / (2.0 * eps)
    );
  }
}

#[test]
fn density_is_the_mixed_derivative_of_the_cdf() {
  let c = Bb7::new(1.5, 1.2);
  let (u, v, eps) = (0.4, 0.7, 1e-4);
  let f = |a: f64, b: f64| c.cdf(array![[a, b]].view()).unwrap()[0];
  let mixed = (f(u + eps, v + eps) - f(u + eps, v - eps) - f(u - eps, v + eps)
    + f(u - eps, v - eps))
    / (4.0 * eps * eps);
  let density = c.pdf(array![[u, v]].view()).unwrap()[0];
  assert!(
    (density - mixed).abs() < 1e-5,
    "density {density} vs fd {mixed}"
  );
}

#[test]
fn cdf_has_uniform_margins_and_nests_the_limits() {
  let c = Bb7::new(1.7, 0.8);
  for (u, v) in [(0.2, 0.7), (0.9, 0.4)] {
    assert!((c.cdf(array![[u, 1.0 - 1e-12]].view()).unwrap()[0] - u).abs() < 1e-8);
    assert!((c.cdf(array![[1.0 - 1e-12, v]].view()).unwrap()[0] - v).abs() < 1e-8);
  }
  // θ = 1 is Clayton(δ).
  let mut clayton = crate::bivariate::clayton::Clayton::default();
  clayton.set_theta(0.8);
  let joe_clayton = Bb7::new(1.0, 0.8);
  let (u, v) = (0.35, 0.65);
  assert!(
    (joe_clayton.cdf(array![[u, v]].view()).unwrap()[0]
      - clayton.cdf(array![[u, v]].view()).unwrap()[0])
      .abs()
      < 1e-12
  );
}

#[test]
fn tail_coefficients_follow_joe() {
  let td = Bb7::new(2.0, 1.0).tail_dependence().unwrap();
  assert!((td.upper - (2.0 - 2.0_f64.sqrt())).abs() < 1e-15);
  assert!((td.lower - 0.5).abs() < 1e-15);
}

/// Cor. 5.1.4's Beta form by mpmath at 60 digits across its pole at `θ = 2`, at `θ = 1` (Clayton), past the switch at
/// `θ = 4` and far out; within `1e-15/min(δ, 1)`, as `R(b) − 1/(1 + b)` cancels to `O(δ)`.
#[test]
fn kendall_tau_matches_the_beta_form() {
  for (theta, delta, want) in [
    (1.6, 1.1, 0.4577513242482392),
    (1.7, 0.9, 0.4463258932897074),
    (2.0, 1.3, 0.529201420029881),
    (2.000000001, 1.3, 0.529201420137199),
    (1.999999999, 1.3, 0.5292014199225632),
    (1.0000001, 0.9, 0.310344851967264),
    (1.25, 0.3, 0.2246470101786293),
    (4.0, 5.0, 0.7567099567099567),
    (4.0001, 0.05, 0.6171694905640738),
    (60.0, 2.0, 0.9680063458487572),
    (1e6, 1.0, 0.999998000004),
  ] {
    let tau = unfitted(delta).tau_from_theta(theta);
    let tolerance = 1e-15 / delta.min(1.0);
    assert!(
      (tau - want).abs() < tolerance,
      "(θ, δ) = ({theta}, {delta}): {tau} vs {want}"
    );
  }
  assert!((unfitted(2.0).tau_from_theta(3.0) - 0.65).abs() < 1e-15);
  assert!((unfitted(0.8).tau_from_theta(1.0) - 0.8 / 2.8).abs() < 1e-15);
  assert_eq!(unfitted(1.1).tau_from_theta(f64::INFINITY), 1.0);
}

#[test]
fn kendall_inversion_and_maximum_likelihood_are_consistent() {
  let truth = Bb7::new(1.6, 1.1);
  let tau = truth.tau_from_theta(1.6);
  let inverted = Bb7::from_tau(tau, 1.1).unwrap();
  assert!(
    (inverted.theta().unwrap() - 1.6).abs() < 1e-9,
    "inverted theta {}",
    inverted.theta().unwrap()
  );
  let sample = truth.sample(4000, &Deterministic::new(13)).unwrap();
  let mut fitted = Bb7::default();
  fitted.fit(sample.view()).unwrap();
  assert!(
    (fitted.theta().unwrap() - 1.6).abs() < 0.25,
    "theta {}",
    fitted.theta().unwrap()
  );
  assert!(
    (fitted.delta() - 1.1).abs() < 0.3,
    "delta {}",
    fitted.delta()
  );
}
