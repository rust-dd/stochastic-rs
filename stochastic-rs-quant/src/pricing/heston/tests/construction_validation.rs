use super::*;

#[test]
#[should_panic(expected = "v0 must satisfy `v0.is_finite() && v0 >= 0.0`, got v0 = -0.01")]
fn new_rejects_negative_v0() {
  let _ = HestonPricer::new(HestonParams { v0: -0.01, ..P });
}

/// A NaN `v0` is caught before it reaches a price that would come back NaN with no origin.
#[test]
#[should_panic(expected = "v0 must satisfy `v0.is_finite() && v0 >= 0.0`, got v0 = NaN")]
fn new_rejects_nan_v0() {
  let _ = HestonPricer::new(HestonParams { v0: f64::NAN, ..P });
}

/// `theta` is a variance, so it is checked like `v0`.
#[test]
#[should_panic(
  expected = "theta must satisfy `theta.is_finite() && theta >= 0.0`, got theta = -0.04"
)]
fn new_rejects_negative_theta() {
  let _ = HestonPricer::new(HestonParams { theta: -0.04, ..P });
}

/// `C` divides by `σ²`: a `sigma == 0` would reach the caller as a NaN price naming no cause.
#[test]
#[should_panic(expected = "sigma must satisfy `sigma.is_finite() && sigma > 0.0`, got sigma = 0.0")]
fn new_rejects_zero_sigma() {
  let _ = HestonPricer::new(HestonParams { sigma: 0.0, ..P });
}

#[test]
#[should_panic(
  expected = "sigma must satisfy `sigma.is_finite() && sigma > 0.0`, got sigma = -0.3"
)]
fn new_rejects_negative_sigma() {
  let _ = HestonPricer::new(HestonParams { sigma: -0.3, ..P });
}

#[test]
#[should_panic(expected = "rho must satisfy `(-1.0..=1.0).contains(&rho)`, got rho = -1.5")]
fn new_rejects_out_of_range_rho() {
  let _ = HestonPricer::new(HestonParams { rho: -1.5, ..P });
}

/// The admissible edges the validation must not swallow. `v0 == 0` is
/// separately relied on by `heston_greeks_nan_at_degenerate_inputs`.
#[test]
fn new_accepts_the_admissible_edges() {
  let zero_variances = HestonPricer::new(HestonParams::new(0.0, 1.5, 0.0, 1e-12, -1.0));
  assert_eq!(zero_variances.params.v0, 0.0);
  assert_eq!(zero_variances.params.theta, 0.0);
  assert_eq!(zero_variances.params.rho, -1.0);
  assert_eq!(
    HestonPricer::new(HestonParams { rho: 1.0, ..P }).params.rho,
    1.0
  );
}

#[test]
fn new_reads_back_exactly_the_parameters_it_took() {
  let p = HestonParams::new(0.0123, 0.7, 0.0456, 0.789, 0.25);
  let m = HestonPricer::new(p);
  assert_eq!(m.params(), p);
  assert_eq!(m.with_vol_risk_premium(0.5).params(), p);
}

/// A zero λ and the default are one model.
#[test]
fn vol_risk_premium_zero_is_the_default() {
  let (s, k, r, q, tau) = (100.0, 105.0, 0.05, 0.02, 0.75);
  assert_eq!(
    HestonPricer::new(P)
      .with_vol_risk_premium(0.0)
      .price_call(s, k, r, q, tau),
    HestonPricer::new(P).price_call(s, k, r, q, tau)
  );
}

/// `violation` is `new`'s first failing check in its panic's words, and a NaN is a violation.
#[test]
fn violation_names_the_first_failing_parameter() {
  assert_eq!(P.violation(), None);
  let nan_rho = HestonParams { rho: f64::NAN, ..P };
  assert_eq!(
    nan_rho.violation().as_deref(),
    Some("rho must satisfy `(-1.0..=1.0).contains(&rho)`, got rho = NaN")
  );
  let two = HestonParams {
    v0: -0.01,
    sigma: 0.0,
    ..P
  };
  assert!(two.violation().unwrap().starts_with("v0 must satisfy"));
  assert!(
    HestonParams {
      kappa: f64::INFINITY,
      ..P
    }
    .violation()
    .is_some()
  );
}

/// Feller stays a warning condition: `2κθ = 0.04` against `σ² = 2.25`
/// violates it, and the model must still construct and price.
#[test]
fn new_does_not_enforce_feller() {
  let m = HestonPricer::new(HestonParams::new(0.04, 0.5, 0.04, 1.5, -0.7));
  let feller = 2.0 * m.params.kappa * m.params.theta;
  assert!(
    feller < m.params.sigma * m.params.sigma,
    "not a Feller violation"
  );
  assert!(m.price_call(100.0, 100.0, 0.05, 0.0, 1.0).is_finite());
}

/// `κθ < 0` drives the variance below zero from `v = 0`; at `θ = 0` any `κ` keeps it non-negative.
#[test]
fn kappa_theta_must_keep_the_drift_at_zero_variance_non_negative() {
  let negative = HestonParams { kappa: -1.5, ..P };
  assert_eq!(
    negative.violation().as_deref(),
    Some("kappa must satisfy `kappa * theta >= 0.0`, got kappa = -1.5")
  );
  let explosive = HestonParams::new(0.04, -1.5, 0.0, 0.3, -0.7);
  assert_eq!(HestonPricer::new(explosive).params.kappa, -1.5);
  assert_eq!(HestonParams::new(0.04, 0.0, 0.04, 0.3, -0.7).kappa, 0.0);
}
