use super::HestonCalibrator;
use super::HestonJacobianMethod;
use super::HestonParams;
use crate::LossMetric;
use crate::OptionType;
use crate::traits::Calibrator;

fn non_feller_params() -> HestonParams {
  HestonParams {
    v0: 0.04,
    kappa: 0.8,
    theta: 0.03,
    sigma: 0.5,
    rho: -0.7,
  }
}

#[test]
fn ordinary_projection_preserves_an_in_range_non_feller_fit() {
  let original = non_feller_params();
  assert!(!original.satisfies_feller_condition());

  let projected = original.clone().projected();

  assert_eq!(projected, original);
  assert!(!projected.satisfies_feller_condition());
}

#[test]
fn ordinary_projection_preserves_vol_of_vol_above_the_legacy_cap() {
  let original = HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.05,
    sigma: 1.0,
    rho: -0.7,
  };

  let projected = original.clone().projected();

  assert_eq!(projected, original);
}

#[test]
fn explicit_feller_projection_remains_available() {
  let original = non_feller_params();

  let projected = original.clone().projected_with_feller_condition();

  assert!(projected.satisfies_feller_condition());
  assert_ne!(projected, original);
}

#[test]
fn exact_non_feller_surface_is_representable_by_the_calibrator() {
  let true_params = non_feller_params();
  let spot = 100.0;
  let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
  let seed = HestonCalibrator::new(
    Some(true_params.clone()),
    vec![1.0; strikes.len()].into(),
    vec![spot; strikes.len()].into(),
    strikes.clone().into(),
    0.02,
    Some(0.01),
    0.5,
    OptionType::Call,
    None,
    None,
    None,
    false,
  );
  let market = seed.compute_model_prices_for_numeric(&true_params);
  let mut calibrator = HestonCalibrator::new(
    Some(true_params),
    market,
    vec![spot; strikes.len()].into(),
    strikes.into(),
    0.02,
    Some(0.01),
    0.5,
    OptionType::Call,
    None,
    None,
    None,
    false,
  );
  calibrator.set_jacobian_method(HestonJacobianMethod::NumericFiniteDiff);

  let result = calibrator.calibrate(None).unwrap();

  assert!(result.converged);
  assert!(!result.params.satisfies_feller_condition());
  assert!(result.loss.get(LossMetric::Rmse) < 1e-8);
}
