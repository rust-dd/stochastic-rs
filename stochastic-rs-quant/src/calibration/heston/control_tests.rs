use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::volatility::HestonPow;
use stochastic_rs_stochastic::volatility::heston::Heston as HestonProcess;

use super::lsq::HestonProblem;
use super::*;
use crate::OptionQuote;
use crate::OptionType;
use crate::QuantError;
use crate::calibration::CalibrationHistory;
use crate::calibration::CalibrationOptions;
use crate::calibration::Regularization;
use crate::pricing::fourier::HestonFourier;
use crate::pricing::heston::HestonParams;
use crate::traits::Calibrator;
use crate::traits::ModelPricer;
use crate::traits::ProcessExt;

const TRUTH: HestonParams = HestonParams {
  v0: 0.04,
  kappa: 1.5,
  theta: 0.04,
  sigma: 0.3,
  rho: -0.6,
};

fn calibrator() -> HestonCalibrator {
  let model = HestonFourier::new(TRUTH);
  let quotes = [0.5, 1.0]
    .into_iter()
    .flat_map(|tau| {
      [90.0, 100.0, 110.0].map(|k| {
        let price = model.price_call(100.0, k, 0.03, 0.01, tau);
        OptionQuote::new(k, tau, price, OptionType::Call)
      })
    })
    .collect();
  HestonCalibrator::new(100.0, 0.03, 0.01, quotes)
}

fn series() -> (Array1<f64>, Array1<f64>) {
  let process = HestonProcess::new(
    100.0,
    0.04,
    2.0,
    0.04,
    0.5,
    -0.6,
    0.0,
    256,
    Some(1.0),
    HestonPow::Sqrt,
    VarianceBoundary::Reflect,
    Deterministic::new(42),
  );
  let [s, v] = process.sample();
  (s, v)
}

#[test]
fn a_start_outside_the_heston_domain_is_an_error_before_the_fit() {
  for (start, field) in [
    (HestonParams { rho: 1.5, ..TRUTH }, "rho"),
    (
      HestonParams {
        v0: f64::NAN,
        ..TRUTH
      },
      "v0",
    ),
  ] {
    let err = calibrator().calibrate(Some(start)).unwrap_err();
    assert!(
      matches!(&err, QuantError::InvalidInput { what: "start", reason } if reason.starts_with(field)),
      "{err}"
    );
  }
  let mut field = calibrator();
  field.start = Some(HestonParams {
    sigma: 0.0,
    ..TRUTH
  });
  assert!(matches!(
    field.calibrate(None),
    Err(QuantError::InvalidInput { what: "start", .. })
  ));
}

/// The trail opens at the start and each later entry is an accepted step, so the weighted
/// objective falls along it to the fit.
#[test]
fn the_trail_records_weighted_market_minus_model() {
  let calibrator = calibrator()
    .with_start(HestonParams {
      v0: 0.06,
      kappa: 2.0,
      theta: 0.05,
      sigma: 0.4,
      rho: -0.3,
    })
    .with_residual_weights(vec![1.0, 2.0, 1.0, 0.5, 1.0, 3.0])
    .with_options(CalibrationOptions::default().with_record_history(true));
  let fit = calibrator.calibrate(None).unwrap();
  let start = HestonProblem::new(&calibrator, calibrator.start.unwrap()).effective_params();
  assert_eq!(fit.history[0].params, start);
  assert_eq!(
    fit.history.last().map(|entry| entry.params),
    Some(fit.params)
  );
  let objective = |entry: &CalibrationHistory<HestonParams>| entry.residuals.dot(&entry.residuals);
  assert!(
    fit.history.len() > 1
      && fit
        .history
        .windows(2)
        .all(|pair| objective(&pair[1]) < objective(&pair[0]))
  );
  let weights = calibrator.residual_weights.as_ref().unwrap();
  for entry in &fit.history {
    let model = calibrator.compute_model_prices_for(&entry.params);
    for (i, quote) in calibrator.quotes.iter().enumerate() {
      let gap = (quote.price - model[i]) * weights[i];
      let residual = entry.residuals[i];
      assert!((residual - gap).abs() < 1e-12, "{residual} against {gap}");
    }
  }
}

#[test]
#[should_panic(expected = "rho must satisfy `(-1.0..=1.0).contains(&rho)`, got rho = 1.5")]
fn with_start_panics_outside_the_heston_domain() {
  let _ = calibrator().with_start(HestonParams { rho: 1.5, ..TRUTH });
}

#[test]
fn a_seed_whose_series_give_no_fit_is_an_error_and_no_seed_starts_from_the_fallback() {
  let seed = HestonMleSeed {
    s: Array1::from_vec(vec![100.0, 101.0]),
    v: Array1::from_vec(vec![0.04, 0.05]),
    ..HestonMleSeed::default()
  };
  let err = calibrator()
    .with_mle_seed(seed)
    .calibrate(None)
    .unwrap_err();
  assert!(
    matches!(
      &err,
      QuantError::InvalidInput {
        what: "heston mle seed",
        ..
      }
    ),
    "{err}"
  );
  let fallback = HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.5,
    rho: -0.5,
  };
  assert_eq!(
    calibrator().calibrate(None).unwrap().params,
    calibrator().calibrate(Some(fallback)).unwrap().params
  );
}

#[test]
fn a_seed_without_a_rate_reads_the_calibrators() {
  let (s, v) = series();
  let seed = HestonMleSeed {
    s,
    v,
    ..HestonMleSeed::default()
  };
  let own = calibrator()
    .with_mle_seed(seed.clone())
    .calibrate(None)
    .unwrap();
  let given = calibrator()
    .with_mle_seed(HestonMleSeed {
      r: Some(0.03),
      ..seed
    })
    .calibrate(None)
    .unwrap();
  assert_eq!(own.params, given.params);
}

#[test]
#[should_panic(expected = "seed.delta must satisfy `seed.delta.is_finite() && seed.delta > 0.0`")]
fn a_seed_with_a_non_positive_step_panics() {
  let _ = calibrator().with_mle_seed(HestonMleSeed {
    delta: Some(0.0),
    ..HestonMleSeed::default()
  });
}

#[test]
fn the_public_fields_are_checked_where_the_builders_panic() {
  let mut weighted = calibrator();
  weighted.residual_weights = Some(Array1::from_elem(2, 1.0));
  assert_eq!(
    weighted.calibrate(None).unwrap_err(),
    QuantError::DimensionMismatch {
      what: "residual weights, one per quote",
      expected: 6,
      got: 2,
    }
  );
  let anchors = Regularization::uniform(vec![0.04; 3], 1.0);
  let err = calibrator()
    .with_options(CalibrationOptions::default().with_regularization(anchors))
    .calibrate(None)
    .unwrap_err();
  assert!(
    matches!(
      err,
      QuantError::DimensionMismatch {
        expected: 5,
        got: 3,
        ..
      }
    ),
    "{err}"
  );
}

#[test]
fn an_iteration_cap_stops_the_fit_and_says_so() {
  let fit = calibrator()
    .with_options(CalibrationOptions::default().with_max_iterations(2))
    .calibrate(None)
    .unwrap();
  assert!(
    !fit.converged && fit.iterations.is_some_and(|n| n <= 2),
    "{:?}",
    fit.iterations
  );
  assert_eq!(fit.message.as_deref(), Some("evaluation budget exhausted"));
}
