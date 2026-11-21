use super::HestonCalibrator;
use super::HestonJacobianMethod;
use crate::LossMetric;
use crate::OptionQuote;
use crate::OptionType;
use crate::pricing::heston::HestonParams;
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

  let projected = original.projected();

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

  let projected = original.projected();

  assert_eq!(projected, original);
}

#[test]
fn explicit_feller_projection_remains_available() {
  let original = non_feller_params();

  let projected = original.projected_with_feller_condition();

  assert!(projected.satisfies_feller_condition());
  assert_ne!(projected, original);
}

#[test]
fn feller_projection_lowers_sigma_alone_past_a_square_root_that_rounds_up() {
  // 2κθ = 2 exactly, and √2 rounded to f64 squares to 2 + 4.4e-16.
  let original = HestonParams {
    v0: 0.04,
    kappa: 5.0,
    theta: 0.2,
    sigma: 2.0,
    rho: -0.7,
  };
  assert!(2.0_f64.sqrt() * 2.0_f64.sqrt() > 2.0);

  let projected = original.projected_with_feller_condition();

  assert!(projected.satisfies_feller_condition());
  assert_eq!(projected.sigma, 2.0_f64.sqrt().next_down());
  assert_eq!(
    projected,
    HestonParams {
      sigma: projected.sigma,
      ..original
    }
  );
  assert_eq!(projected.projected_with_feller_condition(), projected);
}

#[test]
fn exact_non_feller_surface_is_representable_by_the_calibrator() {
  let true_params = non_feller_params();
  let strikes = [80.0, 90.0, 100.0, 110.0, 120.0];
  let quote = |k: f64, price: f64| OptionQuote::new(k, 0.5, price, OptionType::Call);
  let pricer = HestonCalibrator::new(100.0, 0.02, 0.01, strikes.map(|k| quote(k, 1.0)).to_vec());
  let market = pricer.compute_model_prices_for_numeric(&true_params);
  let quotes = strikes
    .iter()
    .zip(&market)
    .map(|(&k, &price)| quote(k, price));
  let calibrator = HestonCalibrator::new(100.0, 0.02, 0.01, quotes.collect())
    .with_start(true_params)
    .with_jacobian_method(HestonJacobianMethod::NumericFiniteDiff);

  let result = calibrator.calibrate(None).unwrap();

  assert!(result.converged);
  assert!(!result.params.satisfies_feller_condition());
  assert!(result.loss.get(LossMetric::Rmse) < 1e-8);
}
