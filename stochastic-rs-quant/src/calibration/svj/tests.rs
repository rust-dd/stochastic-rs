use super::SvjCalibrator;
use super::SvjParams;
use super::loss::bates_call_price;
use crate::LossMetric;
use crate::OptionQuote;
use crate::OptionType;
use crate::QuantError;
use crate::calibration::CalibrationHistory;
use crate::calibration::CalibrationOptions;
use crate::pricing::fourier::BatesFourier;
use crate::traits::CalibrationResult;
use crate::traits::Calibrator;
use crate::traits::ToModel;

const HESTON_REF: [f64; 9] = [
  25.095178, 20.976171, 17.106937, 13.548230, 10.361869, 7.604362, 5.317953, 3.519953, 2.193310,
];
const STRIKES: [f64; 9] = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];
const START: SvjParams = SvjParams {
  v0: 0.06,
  kappa: 2.0,
  theta: 0.06,
  sigma: 0.4,
  rho: -0.5,
  lambda: 0.1,
  mu_j: 0.0,
  sigma_j: 0.1,
};

fn heston_ref_quotes() -> Vec<OptionQuote> {
  STRIKES
    .iter()
    .zip(HESTON_REF)
    .map(|(&k, price)| OptionQuote::new(k, 1.0, price, OptionType::Call))
    .collect()
}

#[test]
fn bates_pricer_matches_heston_reference() {
  let p = SvjParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
    lambda: 0.0,
    mu_j: 0.0,
    sigma_j: 0.01,
  };
  for (i, &k) in STRIKES.iter().enumerate() {
    let price = bates_call_price(&p, 100.0, k, 0.05, 0.0, 1.0);
    assert!(
      (price - HESTON_REF[i]).abs() < 0.1,
      "Heston K={k}: got {price:.6}, expected {:.6}",
      HESTON_REF[i]
    );
  }
}

#[test]
fn svj_calibrate_recovers_heston_prices() {
  let result = SvjCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes())
    .with_start(START)
    .calibrate(None)
    .unwrap();
  assert!(
    result.loss.get(LossMetric::Rmse) < 0.5,
    "SVJ→Heston RMSE={:.6}",
    result.loss.get(LossMetric::Rmse)
  );
}

/// Long-maturity / high-|ρ| regression: `bates_cf` must use the
/// Albrecher-Mayer-Schoutens-Tistaert (2007) "Little Heston Trap" form
/// (`g̃ = 1/g_original`, `exp(-d·τ)`). Original Heston (1993) form develops a
/// branch-cut discontinuity at T = 5y, ρ = -0.9; the Trap form does not.
#[test]
fn svj_bates_little_trap_long_maturity_high_rho() {
  let p = SvjParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.9,
    lambda: 0.3,
    mu_j: -0.05,
    sigma_j: 0.15,
  };
  let call = bates_call_price(&p, 100.0, 100.0, 0.05, 0.0, 5.0);
  assert!(
    call.is_finite() && call > 0.0 && call < 100.0,
    "SVJ bates_cf Trap form: finite positive bounded call required at T=5y, ρ=-0.9, got {call}"
  );
}

/// The trail opens at the projected start and each later entry is an accepted step, so the
/// objective falls along it to the fit.
#[test]
fn the_trail_opens_at_the_start_and_records_market_minus_model() {
  let quotes = heston_ref_quotes();
  let fit = SvjCalibrator::new(100.0, 0.05, 0.0, quotes.clone())
    .with_start(START)
    .with_options(CalibrationOptions::default().with_record_history(true))
    .calibrate(None)
    .unwrap();
  assert!(fit.history.len() > 1, "{} entries", fit.history.len());
  assert_eq!(fit.history[0].params, START.projected());
  assert_eq!(
    fit.history.last().map(|entry| entry.params),
    Some(fit.params)
  );
  let objective = |entry: &CalibrationHistory<SvjParams>| entry.residuals.dot(&entry.residuals);
  assert!(
    fit
      .history
      .windows(2)
      .all(|pair| objective(&pair[1]) < objective(&pair[0]))
  );
  for entry in [&fit.history[0], fit.history.last().unwrap()] {
    for (quote, residual) in quotes.iter().zip(&entry.residuals) {
      let model = bates_call_price(&entry.params, 100.0, quote.strike, 0.05, 0.0, quote.tau);
      assert_eq!(*residual, quote.price - model);
    }
  }
}

#[test]
fn a_start_outside_the_bates_domain_is_an_error() {
  let calibrator = SvjCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes());
  for start in [
    SvjParams {
      sigma: -0.3,
      ..START
    },
    SvjParams {
      rho: f64::NAN,
      ..START
    },
  ] {
    let err = calibrator.calibrate(Some(start)).unwrap_err();
    assert!(
      matches!(err, QuantError::InvalidInput { what: "start", .. }),
      "{err}"
    );
  }
  let mut field = calibrator;
  field.start = Some(SvjParams {
    lambda: -1.0,
    ..START
  });
  assert!(matches!(
    field.calibrate(None),
    Err(QuantError::InvalidInput { what: "start", .. })
  ));
}

#[test]
#[should_panic(expected = "rho must satisfy `(-1.0..=1.0).contains(&rho)`, got rho = 1.5")]
fn with_start_panics_outside_the_bates_domain() {
  let _ = SvjCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes())
    .with_start(SvjParams { rho: 1.5, ..START });
}

#[test]
#[should_panic(expected = "sigma_j must satisfy `sigma_j.is_finite() && sigma_j >= 0.0`")]
fn bates_fourier_new_panics_outside_the_domain() {
  let _ = BatesFourier::new(SvjParams {
    sigma_j: -0.1,
    ..START
  });
}

#[test]
fn the_fit_converts_to_the_bates_model_of_its_parameters() {
  let fit = SvjCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes())
    .with_start(START)
    .with_options(CalibrationOptions::default().with_max_iterations(3))
    .calibrate(None)
    .unwrap();
  assert_eq!(fit.to_model().params(), fit.params);
}

#[test]
fn test_svj_calibrate() {
  let k = [
    80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0,
  ];
  let c_market = [
    21.5, 17.9, 14.2, 11.0, 8.2, 6.0, 4.3, 3.1, 2.2, 1.6, 1.2, 0.9,
  ];
  let quotes = k
    .iter()
    .zip(c_market)
    .map(|(&k, price)| OptionQuote::new(k, 0.5, price, OptionType::Call))
    .collect();
  let result = SvjCalibrator::new(100.0, 0.01, 0.0, quotes)
    .with_start(SvjParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.5,
      rho: -0.7,
      lambda: 0.5,
      mu_j: -0.05,
      sigma_j: 0.1,
    })
    .with_options(CalibrationOptions::default().with_record_history(true))
    .calibrate(None)
    .unwrap();
  assert!(result.rmse().is_finite() && !result.history.is_empty());
}

#[test]
fn the_jacobian_is_the_slope_of_the_residuals_across_the_box() {
  use super::loss::SvjProblem;
  use super::params::*;
  use crate::calibration::least_squares::LeastSquaresProblem;
  use crate::calibration::least_squares::checks::box_points;
  use crate::calibration::least_squares::checks::jacobian_error;
  let calibrator = SvjCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes());
  let mut problem = SvjProblem::new(&calibrator, START);
  let bounds = [
    P_V0, P_KAPPA, P_THETA, P_SIGMA_V, P_RHO, P_LAMBDA, P_MU_J, P_SIGMA_J,
  ];
  let far = ndarray::array![0.2, 12.0, 0.3, 0.9, 0.6, 4.0, -0.3, 0.5];
  for base in [problem.params(), far] {
    for (label, point) in box_points(base.view(), &bounds, &[true; 8]) {
      // Where Feller's projection binds, the stencil runs from the projected point.
      let capped = |k: usize| 2.0 * point[k] * point[k + 1] < point[k + 2].powi(2);
      if [1].into_iter().any(capped) {
        continue;
      }
      let error = jacobian_error(&mut problem, point.view(), |_, x| 2e-3 * x.abs().max(1e-2));
      assert!(error < 2e-2, "{label} of {base}: {error:e}");
    }
  }
}
