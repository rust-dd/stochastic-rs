//! `theta_from_tau` meets the closed forms of Clayton, Gumbel and the Gaussian copula and round-trips Frank's Debye
//! inversion through `tau_from_theta`; a τ at the edge of a family's range is an error, not an infinite θ.

use stochastic_rs_copulas::CopulaError;
use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::bivariate::frank::Frank;
use stochastic_rs_copulas::bivariate::gaussian::GaussianCopula;
use stochastic_rs_copulas::bivariate::gumbel::Gumbel;
use stochastic_rs_copulas::bivariate::independence::Independence;
use stochastic_rs_copulas::traits::BivariateExt;

fn approx_eq(a: f64, b: f64, tol: f64) {
  assert!(
    (a - b).abs() < tol,
    "approx_eq failed: |{a} - {b}| = {} >= {tol}",
    (a - b).abs()
  );
}

fn is_tau_error(result: Result<f64, CopulaError>) -> bool {
  matches!(
    result,
    Err(CopulaError::InvalidParameter { name: "tau", .. })
  )
}

#[test]
fn clayton_theta_from_tau_closed_form() {
  // Clayton: θ = 2τ / (1 - τ).
  for &tau in &[0.1_f64, 0.25, 0.5, 0.75, 0.9] {
    let c = Clayton::new();
    let theta = c.theta_from_tau(tau).unwrap();
    let expected = 2.0 * tau / (1.0 - tau);
    approx_eq(theta, expected, 1e-12);
  }
}

#[test]
fn clayton_tau_one_is_an_error() {
  assert!(is_tau_error(Clayton::new().theta_from_tau(1.0)));
}

#[test]
fn gumbel_theta_from_tau_closed_form() {
  // Gumbel: θ = 1 / (1 - τ), τ ∈ [0, 1).
  for &tau in &[0.0_f64, 0.1, 0.25, 0.5, 0.75, 0.9] {
    let theta = Gumbel::new(None).theta_from_tau(tau).unwrap();
    let expected = 1.0 / (1.0 - tau);
    approx_eq(theta, expected, 1e-12);
  }
}

#[test]
fn gumbel_tau_one_is_an_error() {
  assert!(is_tau_error(Gumbel::new(None).theta_from_tau(1.0)));
}

#[test]
fn frank_theta_zero_for_independence() {
  let theta = Frank::new(None).theta_from_tau(0.0).unwrap();
  approx_eq(theta, 0.0, 1e-9);
}

#[test]
fn frank_theta_sign_matches_tau() {
  // Frank theta has the same sign as tau, opposite sign for negative tau.
  let theta_pos = Frank::new(None).theta_from_tau(0.5).unwrap();
  assert!(
    theta_pos > 0.0,
    "expected positive θ for τ > 0, got {theta_pos}"
  );

  let theta_neg = Frank::new(None).theta_from_tau(-0.5).unwrap();
  assert!(
    theta_neg < 0.0,
    "expected negative θ for τ < 0, got {theta_neg}"
  );
}

#[test]
fn frank_tau_at_the_bounds_is_an_error() {
  for tau in [1.0, -1.0] {
    assert!(
      is_tau_error(Frank::new(None).theta_from_tau(tau)),
      "τ = {tau}"
    );
  }
}

#[test]
fn independence_theta_is_zero() {
  approx_eq(Independence::new().theta_from_tau(0.3).unwrap(), 0.0, 1e-12);
}

/// Frank's τ → θ by Brent and back through the Debye integral: the signs agree and the residual is at Brent's tolerance.
#[test]
fn frank_tau_theta_roundtrip() {
  let frank = Frank::new(None);
  for &tau in &[-0.6_f64, -0.3, 0.1, 0.3, 0.5, 0.7] {
    let theta = frank.theta_from_tau(tau).unwrap();
    assert!(theta.is_finite(), "θ blew up for τ={tau}");
    assert!(
      tau.signum() == theta.signum() || tau.abs() < 1e-9,
      "τ and θ should share sign; τ={tau}, θ={theta}"
    );
    approx_eq(frank.tau_from_theta(theta), tau, 1e-9);
  }
}

#[test]
fn clayton_tau_theta_roundtrip() {
  for &tau in &[0.05_f64, 0.25, 0.5, 0.75, 0.95] {
    let c = Clayton::new();
    let theta = c.theta_from_tau(tau).unwrap();
    let tau_back = theta / (theta + 2.0);
    approx_eq(tau_back, tau, 1e-12);
  }
}

#[test]
fn gumbel_tau_theta_roundtrip() {
  for &tau in &[0.05_f64, 0.25, 0.5, 0.75, 0.95] {
    let theta = Gumbel::new(None).theta_from_tau(tau).unwrap();
    let tau_back = (theta - 1.0) / theta;
    approx_eq(tau_back, tau, 1e-12);
  }
}

/// Gaussian: ρ = sin(πτ/2) (Kendall's τ inversion, Kruskal 1958).
#[test]
fn gaussian_theta_from_tau_closed_form() {
  for &tau in &[-0.6_f64, -0.3, 0.0, 0.25, 0.5, 0.75, 0.9] {
    let c = GaussianCopula::new();
    let theta = c.theta_from_tau(tau).unwrap();
    let expected = (0.5 * std::f64::consts::PI * tau).sin();
    approx_eq(theta, expected, 1e-12);
  }
}
