//! Calibration regression tests and the projection-box pins.

use super::*;
use crate::OptionType;
use crate::calibration::CalibrationHistory;
use crate::calibration::least_squares::checks::jacobian_error;
use crate::traits::CalibrationResult;
use crate::traits::Calibrator;

const TRUE_SIGMA: f64 = 0.25;

fn quotes(sigma: f64, q: f64, cells: &[(f64, f64, OptionType)]) -> Vec<OptionQuote> {
  let pricer = BsmPricer::new(sigma, BsmCoc::Merton1973);
  cells
    .iter()
    .map(|&(k, tau, ot)| {
      let price = pricer.price_option(100.0, k, 0.03, q, tau, ot);
      OptionQuote::new(k, tau, price, ot)
    })
    .collect()
}

/// Calls generated at `TRUE_SIGMA`, with the start written through the public field.
fn synthetic(start: f64) -> BsmCalibrator {
  let cells = [90.0, 95.0, 100.0, 105.0, 110.0].map(|k| (k, 0.5, OptionType::Call));
  let mut calibrator = BsmCalibrator::new(100.0, 0.03, 0.0, quotes(TRUE_SIGMA, 0.0, &cells));
  calibrator.start = BsmParams { sigma: start };
  calibrator
}

#[test]
fn test_calibrate() {
  let k = [
    395.0, 400.0, 405.0, 410.0, 415.0, 420.0, 425.0, 430.0, 435.0, 440.0, 445.0, 450.0,
  ];
  let c_market = [
    30.75, 25.88, 21.00, 16.50, 11.88, 7.69, 4.44, 2.10, 0.78, 0.25, 0.10, 0.10,
  ];
  let quotes = k
    .iter()
    .zip(c_market)
    .map(|(&k, price)| OptionQuote::new(k, 1.0, price, OptionType::Call))
    .collect();
  BsmCalibrator::new(425.73, 0.05, 0.0, quotes)
    .calibrate(None)
    .unwrap();
}

#[test]
fn three_maturities_recover_one_sigma() {
  let cells = [0.10, 0.30, 0.75]
    .into_iter()
    .flat_map(|tau| {
      [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0].map(|k| (k, tau, OptionType::Call))
    })
    .collect::<Vec<_>>();
  let fit = BsmCalibrator::new(100.0, 0.03, 0.0, quotes(TRUE_SIGMA, 0.0, &cells))
    .with_start(BsmParams { sigma: 0.4 })
    .calibrate(None)
    .unwrap();
  assert!(
    (fit.params().sigma - TRUE_SIGMA).abs() < 1e-3,
    "{}",
    fit.params().sigma
  );
  assert!(fit.converged() && fit.max_error().unwrap() < 1e-6);
  assert_eq!(fit.to_model().sigma, fit.params.sigma);
}

#[test]
fn the_dividend_yield_sets_the_carry() {
  let cells = [90.0, 100.0, 110.0].map(|k| (k, 1.0, OptionType::Call));
  let fit = BsmCalibrator::new(100.0, 0.03, 0.02, quotes(TRUE_SIGMA, 0.02, &cells))
    .calibrate(None)
    .unwrap();
  assert!(
    (fit.params.sigma - TRUE_SIGMA).abs() < 1e-10,
    "{}",
    fit.params.sigma
  );
}

/// The trail opens at the start and each later entry is an accepted step, so the vega-weighted
/// objective falls along it to the fit.
#[test]
fn calls_and_puts_fit_together_and_the_trail_records_market_minus_model_over_vega() {
  let cells = [
    (90.0, 0.5, OptionType::Put),
    (100.0, 0.5, OptionType::Call),
    (110.0, 1.0, OptionType::Call),
    (95.0, 1.0, OptionType::Put),
  ];
  let quotes = quotes(TRUE_SIGMA, 0.0, &cells);
  let fit = BsmCalibrator::new(100.0, 0.03, 0.0, quotes.clone())
    .with_options(CalibrationOptions::default().with_record_history(true))
    .calibrate(None)
    .unwrap();
  assert!((fit.params.sigma - TRUE_SIGMA).abs() < 1e-10);
  assert_eq!(fit.history[0].iteration, 0);
  assert_eq!(fit.history[0].params, BsmParams { sigma: 0.2 });
  assert_eq!(
    fit.history.last().map(|entry| entry.params),
    Some(fit.params)
  );
  let objective = |entry: &CalibrationHistory<BsmParams>| entry.residuals.dot(&entry.residuals);
  assert!(
    fit
      .history
      .windows(2)
      .all(|pair| objective(&pair[1]) < objective(&pair[0]))
  );
  for entry in &fit.history {
    let model = BsmPricer::new(entry.params.sigma, BsmCoc::Merton1973);
    for (quote, residual) in quotes.iter().zip(&entry.residuals) {
      let (k, tau) = (quote.strike, quote.tau);
      let gap = quote.price - model.price_option(100.0, k, 0.03, 0.0, tau, quote.option_type);
      let scaled = residual * model.vega(100.0, k, 0.03, 0.0, tau);
      assert!((scaled - gap).abs() < 1e-12, "{scaled} against {gap}");
    }
  }
}

/// basin counts a rejected trial as an iteration and stops a capped fit before its next trial, so
/// a capped fit whose every step is accepted has the start and one entry per iteration.
#[test]
fn the_trail_has_the_start_and_one_entry_per_iteration() {
  let strikes = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];
  let calibrator = BsmCalibrator::new(
    100.0,
    0.03,
    0.0,
    quotes(0.2, 0.0, &strikes.map(|k| (k, 1.0, OptionType::Call))),
  );
  for (start, cap) in [(0.1, 3), (0.5, 2), (0.35, 1)] {
    let options = CalibrationOptions::default()
      .with_record_history(true)
      .with_max_iterations(cap);
    let fit = calibrator
      .clone()
      .with_options(options)
      .calibrate(Some(BsmParams { sigma: start }))
      .unwrap();
    assert_eq!(fit.message.as_deref(), Some("evaluation budget exhausted"));
    assert_eq!(
      (fit.history.len(), fit.iterations),
      (cap + 1, Some(cap)),
      "start {start}"
    );
    assert_eq!(fit.history[0].params, BsmParams { sigma: start });
  }
}

/// The box on the write path, and the number it prevents: a negative `sigma`, built past `new`'s
/// check, prices a finite negative call, so the second assertion tests the projection, not `abs`.
#[test]
fn sigma_is_reflected_not_clamped() {
  let cal = synthetic(0.2);
  let mut problem = BsmProblem::new(&cal, cal.start);
  problem.set_params(Array1::from_vec(vec![-0.3]).view());
  assert_eq!(
    problem.params.sigma, 0.3,
    "a negative step must reflect, not clamp"
  );

  let bsm = |sigma: f64| {
    let b = BsmCoc::Merton1973;
    BsmPricer { sigma, b }.price_call(100.0, 100.0, 0.03, 0.0, 0.5)
  };
  assert!(
    bsm(-0.3).is_finite() && (bsm(0.3) - bsm(-0.3)).abs() > 1.0,
    "an unprojected -0.3 prices {} against {} at +0.3",
    bsm(-0.3),
    bsm(0.3)
  );

  problem.set_params(Array1::from_vec(vec![f64::NAN]).view());
  assert_eq!(
    problem.params.sigma, V_MIN,
    "a projection onto a set must be total"
  );

  let cal = cal.with_start(BsmParams { sigma: -0.4 });
  assert_eq!(
    cal.start.sigma, 0.4,
    "the starting point is stored inside the box"
  );
}

/// The read path: the optimiser prices the start before it has a step to hand back.
#[test]
fn the_optimisers_first_evaluation_is_already_inside_the_box() {
  let cal = synthetic(0.2);
  let first = |sigma: f64| BsmProblem::new(&cal, BsmParams { sigma }).residuals();
  assert_eq!(
    first(-0.4),
    first(0.4),
    "the first residual evaluation must not see the raw sign"
  );
}

/// Started outside the box, by any route in, the fit still lands on the truth.
#[test]
fn a_negative_starting_point_still_recovers_sigma() {
  let start = BsmParams { sigma: -0.4 };
  for (label, result) in [
    ("argument", synthetic(0.2).calibrate(Some(start)).unwrap()),
    ("public field", synthetic(-0.4).calibrate(None).unwrap()),
    (
      "builder",
      synthetic(0.2).with_start(start).calibrate(None).unwrap(),
    ),
  ] {
    assert!(
      (result.params.sigma - TRUE_SIGMA).abs() < 1e-3,
      "{label}: recovered {} against {TRUE_SIGMA}",
      result.params.sigma
    );
  }
}

#[test]
fn a_non_finite_start_is_an_error() {
  for start in [f64::NAN, f64::INFINITY] {
    let err = synthetic(0.2)
      .calibrate(Some(BsmParams { sigma: start }))
      .unwrap_err();
    assert!(
      matches!(&err, QuantError::InvalidInput { what: "start", reason } if reason.starts_with("sigma")),
      "{err}"
    );
    assert!(synthetic(start).calibrate(None).is_err());
  }
}

#[test]
#[should_panic(
  expected = "start.sigma must satisfy `start.sigma.is_finite()`, got start.sigma = NaN"
)]
fn with_start_panics_on_a_non_finite_sigma() {
  let _ = synthetic(0.2).with_start(BsmParams { sigma: f64::NAN });
}

/// At these starts every vega is below `1e-50`, so no residual moves with σ.
#[test]
fn a_start_where_no_quote_has_a_vega_is_not_convergence() {
  for start in [1e-6, 1e-3, 50.0, 1e3, 1e6] {
    let fit = synthetic(start).calibrate(None).unwrap();
    assert!(!fit.converged, "start {start}: {:?}", fit.params);
    assert_eq!(fit.params.sigma, start);
    assert_eq!(fit.message(), Some("flat objective"), "start {start}");
  }
}

/// 19, 20 and 22.4 pass through a reflected iterate, where the Jacobian needs the reflection sign.
#[test]
fn a_start_up_to_ninety_times_the_truth_recovers_sigma() {
  for start in [5.0, 19.0, 20.0, 22.4] {
    let fit = synthetic(start).calibrate(None).unwrap();
    let sigma = fit.params.sigma;
    assert!(
      fit.converged && (sigma - TRUE_SIGMA).abs() < 1e-8,
      "start {start}: {sigma}, {:?}",
      fit.message()
    );
  }
}

#[test]
fn the_jacobian_is_the_slope_of_the_residuals_at_a_reflected_coordinate() {
  let calibrator = synthetic(0.2);
  let mut problem = BsmProblem::new(&calibrator, BsmParams { sigma: 0.2 });
  // The start, the truth, a far point, and each of them reflected.
  for x in [0.2, TRUE_SIGMA, 12.2458, -0.2, -TRUE_SIGMA, -12.2458] {
    let error = jacobian_error(&mut problem, ndarray::array![x].view(), |_, x| {
      1e-6 * x.abs()
    });
    assert!(error < 1e-4, "x = {x}: {error}");
  }
}

/// Below the vega floor the weight is constant, so the row's slope is vega over the floor.
#[test]
fn the_jacobian_is_the_slope_of_the_residuals_on_both_sides_of_the_vega_floor() {
  let calibrator = synthetic(0.2);
  for sigma in [0.2, 0.02, 0.01] {
    let at = |sigma: f64| {
      let mut problem = BsmProblem::new(&calibrator, BsmParams { sigma });
      (problem.residuals().unwrap(), problem.jacobian().unwrap())
    };
    let h = 1e-6 * sigma;
    let slope = (at(sigma + h).0 - at(sigma - h).0) / (2.0 * h);
    let (residuals, jacobian) = at(sigma);
    for (row, expected) in slope.iter().enumerate() {
      let scale = expected.abs().max(residuals[row].abs() * 1e-9).max(1e-12);
      assert!(
        (jacobian[(row, 0)] - expected).abs() < 1e-4 * scale,
        "sigma {sigma}, row {row}: {} vs {expected}",
        jacobian[(row, 0)]
      );
    }
  }
}
