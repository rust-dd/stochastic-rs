use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::LossMetric;
use crate::QuantError;
use crate::calibration::CalibrationOptions;
use crate::traits::CalibrationResult;
use crate::traits::Calibrator;

#[test]
fn test_empirical_wasserstein_1_matches_simple_case() {
  let a = vec![1.0, 2.0, 3.0];
  let b = vec![2.0, 3.0, 4.0];
  let w1 = empirical_wasserstein_1(&a, &b);
  assert!((w1 - 1.0).abs() < 1e-12);
}

fn slices(maturities: &[f64]) -> Vec<RBergomiMarketSlice> {
  let truth = RBergomiParams {
    hurst: 0.12,
    rho: -0.72,
    eta: 1.6,
    xi0: RBergomiXi0::Constant(0.04),
  };
  let slice = |(i, &tau): (usize, &f64)| RBergomiMarketSlice {
    tau,
    terminal_samples: simulate_rbergomi_terminal_samples(
      &truth,
      100.0,
      0.01,
      0.0,
      tau,
      512,
      96,
      12,
      &Deterministic::new(7_777 + i as u64),
    ),
  };
  maturities.iter().enumerate().map(slice).collect()
}

fn start() -> RBergomiParams {
  RBergomiParams {
    hurst: 0.30,
    rho: -0.10,
    eta: 0.80,
    xi0: RBergomiXi0::Constant(0.02),
  }
}

fn config() -> RBergomiCalibrationConfig {
  RBergomiCalibrationConfig {
    paths: 512,
    steps_per_year: 96,
    msoe_terms: 12,
    learning_rate: 0.08,
    finite_diff_eps: 5e-3,
    adam_beta1: 0.9,
    adam_beta2: 0.99,
    adam_eps: 1e-8,
    stop_loss: None,
    improvement_tol: 1e-5,
  }
}

#[test]
fn test_rbergomi_calibration_reduces_loss_on_synthetic_data() {
  let options = CalibrationOptions::default()
    .with_max_iterations(12)
    .with_record_history(true);
  let calibrator = RBergomiCalibrator::new(
    100.0,
    0.01,
    0.0,
    start(),
    slices(&[0.25, 0.5, 1.0]),
    config(),
  )
  .with_seed(Deterministic::new(123_456))
  .with_options(options);
  let fit = calibrator.calibrate(None).unwrap();
  let first = &fit.history[0];
  assert_eq!(first.iteration, 0);
  assert_eq!(first.params.hurst, start().projected().hurst);
  assert_eq!(first.residuals.len(), 3);
  let initial = first.loss.get(LossMetric::Mse);
  let w1 = fit.loss.get(LossMetric::Mse);
  assert!(w1 <= initial, "W1 {w1} against {initial} at the start");
  assert_eq!(fit.rmse(), w1.abs().sqrt());
  assert!(fit.iterations.is_some_and(|n| n <= 12));
  assert!(fit.params.hurst > 0.0 && fit.params.hurst < 0.5);
  assert!(fit.params.rho.abs() < 1.0 && fit.params.eta > 0.0);
  assert!(matches!(fit.params.xi0, RBergomiXi0::Constant(x) if x > 0.0));
  let replay = calibrator.calibrate(None).unwrap();
  assert_eq!(
    replay.history.len(),
    fit.history.len(),
    "one trail per call"
  );
}

#[test]
fn the_start_and_the_slices_are_data() {
  let calibrator = RBergomiCalibrator::new(100.0, 0.01, 0.0, start(), slices(&[0.5]), config());
  let bad_start = RBergomiParams {
    hurst: 0.7,
    ..start()
  };
  let err = calibrator.calibrate(Some(bad_start)).unwrap_err();
  assert!(
    matches!(
      err,
      QuantError::InvalidInput {
        what: "RBergomiParams.hurst",
        ..
      }
    ),
    "{err}"
  );
  let mut empty = calibrator.clone();
  empty.market_slices.clear();
  assert_eq!(
    empty.calibrate(None).unwrap_err(),
    QuantError::InsufficientData {
      what: "market slices",
      needed: 1,
      got: 0
    }
  );
  let mut nan = calibrator;
  nan.market_slices[0].terminal_samples[3] = f64::NAN;
  assert!(matches!(
    nan.calibrate(None).unwrap_err(),
    QuantError::InvalidInput {
      what: "RBergomiMarketSlice.terminal_samples",
      ..
    }
  ));
}

/// One short slice on few paths: a stop needs a step or two, not a converged fit.
fn quick(config: RBergomiCalibrationConfig) -> RBergomiCalibrator {
  let config = RBergomiCalibrationConfig {
    paths: 256,
    steps_per_year: 48,
    ..config
  };
  RBergomiCalibrator::new(100.0, 0.01, 0.0, start(), slices(&[0.25]), config)
}

#[test]
fn converged_is_a_stop_on_the_fits_own_criterion_and_the_message_names_it() {
  let loose = RBergomiCalibrationConfig {
    improvement_tol: 1e9,
    ..config()
  };
  let fit = quick(loose).calibrate(None).unwrap();
  assert!(fit.converged, "{:?}", fit.message);
  assert_eq!(
    fit.message.as_deref(),
    Some("improvement below improvement_tol")
  );
  let reached = RBergomiCalibrationConfig {
    stop_loss: Some(1e9),
    ..config()
  };
  let fit = quick(reached).calibrate(None).unwrap();
  assert!(fit.converged, "{:?}", fit.message);
  assert_eq!(fit.message.as_deref(), Some("loss reached stop_loss"));
  let strict = RBergomiCalibrationConfig {
    improvement_tol: 0.0,
    ..config()
  };
  let capped = CalibrationOptions::default().with_max_iterations(1);
  let fit = quick(strict).with_options(capped).calibrate(None).unwrap();
  assert!(!fit.converged, "{:?}", fit.message);
  assert_eq!(fit.message.as_deref(), Some("iteration cap reached"));
}

#[test]
fn a_knob_adam_cannot_read_is_unsupported() {
  let capped = || CalibrationOptions::default().with_max_iterations(1);
  let pull = crate::calibration::Regularization::uniform(vec![0.0; 4], 1.0);
  for (options, what) in [
    (capped().with_tolerance(1e-6), "tolerance"),
    (capped().with_patience(5), "patience"),
    (
      capped().with_loss_metrics(&[LossMetric::Mae]),
      "loss_metrics",
    ),
    (capped().with_regularization(pull), "regularization"),
  ] {
    let err = quick(config())
      .with_options(options)
      .calibrate(None)
      .unwrap_err();
    assert_eq!(err, QuantError::Unsupported { what });
  }
}

#[test]
#[should_panic(expected = "config.paths must satisfy `config.paths > 0`, got config.paths = 0")]
fn a_zero_path_count_panics() {
  let config = RBergomiCalibrationConfig {
    paths: 0,
    ..config()
  };
  let _ = RBergomiCalibrator::new(100.0, 0.01, 0.0, start(), Vec::new(), config);
}

/// Reference: `scipy.stats.wasserstein_distance` 1.17, the integral of `|F⁻¹ − G⁻¹|` over the two
/// empirical quantile functions; for equal sizes it is the module doc's order-statistic mean.
#[test]
fn wasserstein_1_is_the_distance_between_the_two_empirical_measures() {
  for (x, y, want) in [
    (vec![0.0, 10.0], vec![1.0, 9.0], 1.0),
    (
      vec![3.0, -1.0, 4.0, 1.5, 9.0],
      vec![2.0, 7.0, 1.0, 8.0, -2.5],
      1.4,
    ),
    (vec![0.0, 1.0, 3.0], vec![5.0, 6.0], 4.166666666666667),
    (
      vec![1.0, 2.0, 4.0, 8.0],
      vec![0.5, 1.5, 2.5, 3.5, 5.0, 13.0],
      1.5,
    ),
    (vec![2.0], vec![1.0, 2.0, 6.0], 1.6666666666666667),
  ] {
    let (got, swapped) = (
      empirical_wasserstein_1(&x, &y),
      empirical_wasserstein_1(&y, &x),
    );
    assert!((got - want).abs() < 1e-14, "{x:?}, {y:?}: {got} vs {want}");
    assert!((swapped - want).abs() < 1e-14, "{y:?}, {x:?}: {swapped}");
  }
}

#[test]
fn wasserstein_1_is_nan_for_a_sample_that_is_not_finite_or_is_missing() {
  let finite = [1.0, 9.0];
  for bad in [
    vec![0.0, f64::NAN, 10.0],
    vec![0.0, -f64::NAN, 10.0],
    vec![0.0, f64::INFINITY],
    vec![f64::NEG_INFINITY, 0.0],
    vec![],
  ] {
    assert!(empirical_wasserstein_1(&bad, &finite).is_nan(), "{bad:?}");
    assert!(empirical_wasserstein_1(&finite, &bad).is_nan(), "{bad:?}");
  }
}

#[test]
fn the_calibrator_holds_its_slices_in_maturity_order() {
  let calibrator = RBergomiCalibrator::new(
    100.0,
    0.01,
    0.0,
    start(),
    slices(&[0.5, 0.25, 1.0]),
    config(),
  );
  let taus = calibrator
    .market_slices()
    .iter()
    .map(|slice| slice.tau)
    .collect::<Vec<f64>>();
  assert_eq!(taus, [0.25, 0.5, 1.0]);
}
