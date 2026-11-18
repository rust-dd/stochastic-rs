use super::*;
use crate::traits::CalibrationResult;
use crate::traits::Calibrator;
use crate::traits::ModelPricer;

const TRUTH: SabrParams = SabrParams {
  alpha: 0.035,
  beta: 0.5,
  rho: -0.25,
  nu: 0.45,
};

fn start(beta: f64, shift: f64) -> SabrCapletParams {
  SabrCapletParams {
    sabr: SabrParams {
      alpha: 0.03,
      beta,
      rho: -0.1,
      nu: 0.3,
    },
    shift,
  }
}

fn smile() -> SabrCapletCalibrator {
  let strikes: Vec<f64> = (0..=10).map(|i| 0.02 + 0.004 * i as f64).collect();
  let vols = strikes
    .iter()
    .map(|&k| hagan_implied_vol(k, 0.04, 2.0, &TRUTH))
    .collect();
  SabrCapletCalibrator::new(0.04, 2.0, strikes, vols, start(0.5, 0.0))
}

#[test]
fn the_fit_recovers_the_smile_and_its_recorded_trail_ends_at_it() {
  let options = CalibrationOptions::default()
    .with_max_iterations(800)
    .with_tolerance(1e-12)
    .with_record_history(true);
  let fit = smile().with_options(options).calibrate(None).unwrap();
  let p = fit.params.sabr;
  assert!(fit.converged && fit.rmse() < 1e-6, "rmse {}", fit.rmse());
  assert!((p.alpha - 0.035).abs() < 1e-3 && (p.nu - 0.45).abs() < 1e-2 && p.beta == 0.5);
  assert!(fit.max_error.unwrap() < 1e-5);
  assert_eq!(fit.history.len(), fit.iterations.unwrap() + 1);
  assert_eq!(fit.history.last().unwrap().params.sabr, p);
  let vol = fit.to_shifted_volatility();
  let model = fit.to_model();
  assert!(vol.alpha == p.alpha && vol.shift == 0.0 && model.params() == p);
  assert!(model.price_call(0.04, 0.04, 0.0, 0.0, 2.0).is_finite());
}

#[test]
fn a_start_replaces_beta_and_the_shift() {
  let fit = smile().calibrate(Some(start(1.0, 0.01))).unwrap();
  assert_eq!((fit.params.sabr.beta, fit.params.shift), (1.0, 0.01));
  assert_eq!(fit.to_shifted_volatility().shift, 0.01);
}

#[test]
fn every_bad_datum_is_its_error() {
  let forward = |forward: f64, strikes: Vec<f64>, shift: f64| {
    let vols = vec![0.5; strikes.len()];
    SabrCapletCalibrator::new(forward, 0.5, strikes, vols, start(1.0, shift))
      .calibrate(None)
      .unwrap_err()
  };
  let err = forward(-0.002, vec![-0.005, 0.0, 0.005], 0.0);
  assert!(
    matches!(&err, QuantError::InvalidInput { what: "forward", reason } if reason.contains("-0.002")),
    "{err}"
  );
  let err = forward(-0.002, vec![-0.005, 0.0, 0.005], 0.003);
  assert!(
    matches!(&err, QuantError::InvalidQuote { index: 0, reason } if reason.contains("-0.005")),
    "{err}"
  );
  assert!(matches!(
    forward(0.04, Vec::new(), 0.0),
    QuantError::InsufficientData { got: 0, .. }
  ));
  let mut nan_vol = smile();
  nan_vol.market_vols[3] = f64::NAN;
  assert!(matches!(
    nan_vol.calibrate(None).unwrap_err(),
    QuantError::InvalidQuote { index: 3, .. }
  ));
  let mut no_time = smile();
  no_time.tau = 0.0;
  assert!(matches!(
    no_time.calibrate(None).unwrap_err(),
    QuantError::InvalidInput { what: "tau", .. }
  ));
  let err = smile().calibrate(Some(start(1.5, 0.0))).unwrap_err();
  assert!(
    matches!(&err, QuantError::InvalidInput { what: "start", reason } if reason.starts_with("beta")),
    "{err}"
  );
  let anchors =
    CalibrationOptions::default().with_regularization(Regularization::uniform(vec![0.0], 1.0));
  assert!(matches!(
    smile().with_options(anchors).calibrate(None).unwrap_err(),
    QuantError::DimensionMismatch {
      expected: 3,
      got: 1,
      ..
    }
  ));
  let patience = CalibrationOptions::default().with_patience(5);
  assert_eq!(
    smile().with_options(patience).calibrate(None).unwrap_err(),
    QuantError::Unsupported { what: "patience" }
  );
}

#[test]
fn an_anchor_on_nu_pulls_the_fit() {
  let anchor = Regularization::new(vec![0.035, -0.25, 0.9], vec![0.0, 0.0, 1.0]);
  let options = CalibrationOptions::default().with_regularization(anchor);
  let pulled = smile().with_options(options).calibrate(None).unwrap();
  assert!(pulled.params.sabr.nu > 0.6, "nu {}", pulled.params.sabr.nu);
}

#[test]
#[should_panic(expected = "beta must satisfy `(0.0..=1.0).contains(&beta)`, got beta = 1.5")]
fn new_panics_on_a_beta_outside_the_unit_interval() {
  let _ = SabrCapletCalibrator::new(0.04, 2.0, vec![0.04], vec![0.2], start(1.5, 0.0));
}

#[test]
#[should_panic(expected = "weights must satisfy `weights.len() == strikes.len()`")]
fn one_weight_per_strike() {
  let _ = smile().with_weights(vec![1.0; 3]);
}

#[test]
fn a_non_finite_vertex_costs_nan() {
  let calibrator = smile();
  let cost = SabrCapletCost {
    forward: calibrator.forward,
    shift: 0.0,
    tau: calibrator.tau,
    beta: 0.5,
    strikes: calibrator.strikes.clone(),
    market_vols: calibrator.market_vols.clone(),
    weights: vec![1.0; calibrator.strikes.len()],
    regularization: None,
  };
  assert!(cost.cost(&vec![0.03, 0.3, -0.1]).unwrap().is_finite());
  for (axis, bad) in [
    (0, f64::NAN),
    (1, f64::NAN),
    (2, f64::NAN),
    (0, f64::INFINITY),
  ] {
    let mut x = vec![0.03, 0.3, -0.1];
    x[axis] = bad;
    assert!(cost.cost(&x).unwrap().is_nan(), "axis {axis}: {bad}");
    let vols = cost.model_vols(&cost.params(&x));
    assert!(vols.iter().all(|v| v.is_nan()), "axis {axis}: {bad}");
  }
}

#[test]
fn a_nan_strike_forward_or_weight_never_starts_a_fit() {
  let mut nan_strike = smile();
  nan_strike.strikes[2] = f64::NAN;
  assert!(matches!(
    nan_strike.calibrate(None).unwrap_err(),
    QuantError::InvalidQuote { index: 2, .. }
  ));
  let mut nan_forward = smile();
  nan_forward.forward = f64::NAN;
  assert!(matches!(
    nan_forward.calibrate(None).unwrap_err(),
    QuantError::InvalidInput {
      what: "forward",
      ..
    }
  ));
  let mut nan_weight = smile();
  nan_weight.weights = Some(vec![f64::NAN; nan_weight.strikes.len()]);
  assert!(nan_weight.calibrate(None).is_ok_and(|fit| !fit.converged));
}
