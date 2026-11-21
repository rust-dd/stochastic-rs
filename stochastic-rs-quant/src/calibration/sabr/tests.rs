use super::*;
use crate::OptionType;
use crate::calibration::CalibrationHistory;
use crate::traits::CalibrationResult;
use crate::traits::Calibrator;
use crate::traits::ModelPricer;

fn smile(truth: SabrParams, r: f64, q: f64, tau: f64) -> Vec<OptionQuote> {
  [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0]
    .into_iter()
    .map(|k| {
      let price = SabrPricer::new(truth).price_call(100.0, k, r, q, tau);
      OptionQuote::new(k, tau, price, OptionType::Call)
    })
    .collect()
}

#[test]
fn sabr_params_array_round_trip_preserves_beta() {
  for &beta in &[0.0, 0.25, 0.5, 0.75, 1.0] {
    let p = SabrParams {
      alpha: 0.18,
      beta,
      rho: -0.31,
      nu: 0.62,
    };
    let v = Array1::from(p);
    let p2 = SabrParams::from(v);
    assert!((p.alpha - p2.alpha).abs() < 1e-15);
    assert!(
      (p.beta - p2.beta).abs() < 1e-15,
      "β must round-trip: input {beta}, got {}",
      p2.beta
    );
    assert!((p.nu - p2.nu).abs() < 1e-15);
    assert!((p.rho - p2.rho).abs() < 1e-15);
  }
}

#[test]
fn a_recorded_fit_recovers_the_smile_and_its_trail_is_per_call() {
  let truth = SabrParams {
    alpha: 0.2,
    beta: 1.0,
    rho: -0.4,
    nu: 0.6,
  };
  let quotes = smile(truth, 0.02, 0.01, 0.5);
  let calibrator = SabrCalibrator::new(100.0, 0.02, 0.01, quotes.clone())
    .with_start(SabrParams {
      alpha: 0.15,
      beta: 1.0,
      rho: 0.0,
      nu: 0.8,
    })
    .with_options(CalibrationOptions::default().with_record_history(true));
  let fit = calibrator.calibrate(None).unwrap();
  assert!(fit.converged() && fit.rmse() < 1e-6, "rmse {}", fit.rmse());
  assert!((fit.params.nu - 0.6).abs() < 1e-3 && (fit.params.rho + 0.4).abs() < 1e-3);
  assert_eq!(fit.message(), Some("converged"));
  assert!(fit.iterations.is_some() && fit.max_error.unwrap() < 1e-5);
  assert!(fit.history[0].iteration == 0 && fit.history[0].params.alpha == 0.15);
  assert_eq!(
    fit.history.last().map(|entry| entry.params),
    Some(fit.params)
  );
  let objective = |entry: &CalibrationHistory<SabrParams>| entry.residuals.dot(&entry.residuals);
  assert!(
    fit
      .history
      .windows(2)
      .all(|pair| objective(&pair[1]) < objective(&pair[0]))
  );
  for entry in &fit.history {
    let model = SabrPricer::new(entry.params);
    for (quote, residual) in quotes.iter().zip(&entry.residuals) {
      let gap = quote.price - model.price_call(100.0, quote.strike, 0.02, 0.01, quote.tau);
      assert!((residual - gap).abs() < 1e-12, "{residual} against {gap}");
    }
  }
  let again = calibrator.calibrate(None).unwrap();
  assert_eq!(again.history.len(), fit.history.len());
  assert_eq!(again.params, fit.params);
}

#[test]
fn a_bad_strike_is_an_invalid_quote_and_a_bad_start_an_invalid_input() {
  for bad_k in [0.0, -10.0, f64::NAN] {
    let quotes = vec![OptionQuote::new(bad_k, 0.5, 1.0, OptionType::Call)];
    let err = SabrCalibrator::new(100.0, 0.02, 0.01, quotes)
      .calibrate(None)
      .unwrap_err();
    assert!(
      matches!(&err, QuantError::InvalidQuote { index: 0, reason } if reason.starts_with("strike")),
      "k = {bad_k}: {err}"
    );
  }
  let quotes = vec![OptionQuote::new(100.0, 0.5, 1.0, OptionType::Call)];
  let calibrator = SabrCalibrator::new(100.0, 0.02, 0.01, quotes);
  let bad_start = SabrParams {
    alpha: 0.2,
    beta: 1.5,
    rho: 0.0,
    nu: 0.5,
  };
  let err = calibrator.calibrate(Some(bad_start)).unwrap_err();
  assert!(
    matches!(&err, QuantError::InvalidInput { what: "start", reason } if reason.starts_with("beta")),
    "{err}"
  );
  assert!(matches!(
    SabrCalibrator::new(100.0, 0.02, 0.01, Vec::new())
      .calibrate(None)
      .unwrap_err(),
    QuantError::InsufficientData { .. }
  ));
}

#[test]
#[should_panic(expected = "s must satisfy `s.is_finite() && s > 0.0`, got s = 0.0")]
fn a_non_positive_spot_panics_in_new() {
  let _ = SabrCalibrator::new(0.0, 0.02, 0.01, Vec::new());
}

#[test]
#[should_panic(expected = "beta must satisfy `(0.0..=1.0).contains(&beta)`, got beta = 1.5")]
fn a_start_outside_the_parameter_space_panics() {
  let _ = SabrCalibrator::new(100.0, 0.02, 0.01, Vec::new()).with_start(SabrParams {
    alpha: 0.2,
    beta: 1.5,
    rho: 0.0,
    nu: 0.5,
  });
}

#[test]
fn a_put_quote_is_fitted_as_a_put() {
  let truth = SabrParams {
    alpha: 0.25,
    beta: 1.0,
    rho: -0.3,
    nu: 0.5,
  };
  let quotes: Vec<OptionQuote> = [85.0, 95.0, 100.0, 105.0, 115.0]
    .into_iter()
    .map(|k| {
      let pricer = SabrPricer::new(truth);
      let (option_type, price) = if k < 100.0 {
        (OptionType::Put, pricer.price_put(100.0, k, 0.01, 0.0, 1.0))
      } else {
        (
          OptionType::Call,
          pricer.price_call(100.0, k, 0.01, 0.0, 1.0),
        )
      };
      OptionQuote::new(k, 1.0, price, option_type)
    })
    .collect();
  let fit = SabrCalibrator::new(100.0, 0.01, 0.0, quotes)
    .calibrate(None)
    .unwrap();
  assert!(
    fit.rmse() < 1e-6 && (fit.params.nu - 0.5).abs() < 1e-3,
    "{:?}",
    fit.params
  );
}

#[test]
fn a_start_written_through_the_field_is_validated_like_an_explicit_one() {
  let quotes = vec![OptionQuote::new(100.0, 0.5, 1.0, OptionType::Call)];
  let mut calibrator = SabrCalibrator::new(100.0, 0.02, 0.01, quotes);
  calibrator.start = Some(SabrParams {
    alpha: 0.2,
    beta: 1.5,
    rho: 0.0,
    nu: 0.5,
  });
  let err = calibrator.calibrate(None).unwrap_err();
  assert!(
    matches!(&err, QuantError::InvalidInput { what: "start", reason } if reason.starts_with("beta")),
    "{err}"
  );
}

#[test]
fn the_jacobian_is_the_slope_of_the_residuals_at_reflected_coordinates() {
  use crate::calibration::least_squares::checks::jacobian_error;
  let truth = SabrParams {
    alpha: 0.2,
    beta: 1.0,
    rho: -0.4,
    nu: 0.6,
  };
  let calibrator = SabrCalibrator::new(100.0, 0.02, 0.01, smile(truth, 0.02, 0.01, 0.5));
  let mut problem = SabrProblem {
    calibrator: &calibrator,
    market: calibrator.quotes.iter().map(|quote| quote.price).collect(),
    forwards: vec![forward_fx(100.0, 0.5, 0.02, 0.01); calibrator.quotes.len()],
    params: DEFAULT_START,
    reflections: Array1::ones(3),
    trail: Trail::new(),
  };
  // `(α, ν, ρ)`: the start, the truth, a far point, each reflected in α and ν, beside ρ's bounds.
  let points = [
    DEFAULT_START.as_lm_vec(),
    truth.as_lm_vec(),
    ndarray::array![1.5, 3.0, 0.7],
    ndarray::array![-0.2, 0.6, -0.4],
    ndarray::array![0.2, -0.6, -0.4],
    ndarray::array![-1.5, -3.0, 0.7],
    ndarray::array![0.2, 0.6, -0.98],
    ndarray::array![0.2, 0.6, 0.98],
  ];
  let beside = [0.9999 - 0.3e-3 * 0.9999, -(0.9999 - 0.3e-3 * 0.9999)];
  for rho in beside {
    let point = ndarray::array![0.2, 0.6, rho];
    let error = jacobian_error(&mut problem, point.view(), |_, x| 1e-4 * x.abs().max(0.1));
    assert!(error < 1e-2, "{point}: {error:e}");
  }
  for point in points {
    let error = jacobian_error(&mut problem, point.view(), |_, x| 2e-3 * x.abs().max(0.1));
    assert!(error < 2e-2, "{point}: {error:e}");
  }
}

#[test]
fn a_non_finite_parameter_reaches_the_residuals_as_nan() {
  let truth = SabrParams {
    alpha: 0.2,
    beta: 1.0,
    rho: -0.4,
    nu: 0.6,
  };
  let calibrator = SabrCalibrator::new(100.0, 0.02, 0.01, smile(truth, 0.02, 0.01, 0.5));
  let mut problem = SabrProblem {
    calibrator: &calibrator,
    market: calibrator.quotes.iter().map(|quote| quote.price).collect(),
    forwards: vec![forward_fx(100.0, 0.5, 0.02, 0.01); calibrator.quotes.len()],
    params: DEFAULT_START,
    reflections: Array1::ones(3),
    trail: Trail::new(),
  };
  for (axis, bad) in [
    (0, f64::NAN),
    (1, f64::NAN),
    (2, f64::NAN),
    (0, f64::INFINITY),
  ] {
    let mut x = DEFAULT_START.as_lm_vec();
    x[axis] = bad;
    problem.set_params(x.view());
    assert!(!problem.params()[axis].is_finite(), "axis {axis}: {bad}");
    let residuals = problem.residuals().unwrap();
    assert!(residuals.iter().all(|r| r.is_nan()), "axis {axis}: {bad}");
    let jacobian = problem.jacobian().unwrap();
    assert!(jacobian.iter().any(|d| d.is_nan()), "axis {axis}: {bad}");
  }
}

#[test]
fn the_projection_keeps_a_nan() {
  for axis in 0..4 {
    let mut v = Array1::from(DEFAULT_START);
    v[axis] = f64::NAN;
    let projected = Array1::from(SabrParams::from(v).projected());
    assert!(projected[axis].is_nan(), "axis {axis}: {projected}");
  }
}

#[test]
fn a_nan_price_or_maturity_is_an_invalid_quote() {
  for (tau, price) in [(f64::NAN, 1.0), (0.5, f64::NAN)] {
    let quotes = vec![
      OptionQuote::new(100.0, 0.5, 1.0, OptionType::Call),
      OptionQuote::new(100.0, tau, price, OptionType::Call),
    ];
    let err = SabrCalibrator::new(100.0, 0.02, 0.01, quotes)
      .calibrate(None)
      .unwrap_err();
    assert!(
      matches!(err, QuantError::InvalidQuote { index: 1, .. }),
      "{err}"
    );
  }
}
