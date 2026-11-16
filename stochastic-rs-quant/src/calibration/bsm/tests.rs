//! Calibration regression tests and the projection-box pins.

use super::*;
use crate::OptionType;
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

#[test]
fn calls_and_puts_fit_together_and_the_trail_is_recorded() {
  let cells = [
    (90.0, 0.5, OptionType::Put),
    (100.0, 0.5, OptionType::Call),
    (110.0, 1.0, OptionType::Call),
    (95.0, 1.0, OptionType::Put),
  ];
  let fit = BsmCalibrator::new(100.0, 0.03, 0.0, quotes(TRUE_SIGMA, 0.0, &cells))
    .with_options(CalibrationOptions::default().with_record_history(true))
    .calibrate(None)
    .unwrap();
  assert!((fit.params.sigma - TRUE_SIGMA).abs() < 1e-10);
  assert_eq!(fit.history[0].iteration, 0);
  assert_eq!(fit.history[0].params, BsmParams { sigma: 0.2 });
  assert!(fit.history.len() > 1);
}

/// The box on the write path, and the number it prevents: a negative `sigma` prices a finite,
/// here negative, call, so the second assertion tests the projection rather than `abs`.
#[test]
fn sigma_is_reflected_not_clamped() {
  let mut cal = synthetic(0.2);
  LeastSquaresProblem::set_params(&mut cal, &Array1::from_vec(vec![-0.3]));
  assert_eq!(
    cal.start.sigma, 0.3,
    "a negative step must reflect, not clamp"
  );

  let bsm = |v: f64| BsmPricer::new(v, BsmCoc::Merton1973).price_call(100.0, 100.0, 0.03, 0.0, 0.5);
  assert!(
    bsm(-0.3).is_finite() && (bsm(0.3) - bsm(-0.3)).abs() > 1.0,
    "an unprojected -0.3 prices {} against {} at +0.3",
    bsm(-0.3),
    bsm(0.3)
  );

  LeastSquaresProblem::set_params(&mut cal, &Array1::from_vec(vec![f64::NAN]));
  assert_eq!(
    cal.start.sigma, V_MIN,
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
  assert_eq!(
    LeastSquaresProblem::residuals(&mut synthetic(-0.4)),
    LeastSquaresProblem::residuals(&mut synthetic(0.4)),
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
