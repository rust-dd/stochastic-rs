use super::LevyCalibrator;
use super::loss::fourier_call_price;
use super::loss::fourier_option_price;
use super::types::LevyModel;
use super::types::LevyModelType;
use super::types::LevyParams;
use crate::CalibrationLossScore;
use crate::LossMetric;
use crate::OptionQuote;
use crate::OptionType;
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationHistory;
use crate::calibration::CalibrationOptions;
use crate::pricing::fourier::VarianceGammaFourier;
use crate::traits::CalibrationResult;
use crate::traits::Calibrator;
use crate::traits::ModelPricer;
use crate::traits::ToModel;

// Analytical reference prices (Gil-Pelaez inversion)
// S=100, r=0.05, q=0, T=1.0
const STRIKES: [f64; 9] = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];

// Vg: sigma=0.2, theta=-0.1, nu=0.5
const VG_REF: [f64; 9] = [
  25.056158, 20.941767, 17.091301, 13.572373, 10.453503, 7.795810, 5.640544, 3.991334, 2.793823,
];

// MJD: sigma=0.15, lam=1.0, muj=-0.05, sigj=0.1
const MJD_REF: [f64; 9] = [
  24.537096, 20.322713, 16.420092, 12.915553, 9.877019, 7.340385, 5.303578, 3.729825, 2.557790,
];

fn calls(strikes: &[f64], prices: &[f64], tau: f64) -> Vec<OptionQuote> {
  strikes
    .iter()
    .zip(prices)
    .map(|(&k, &price)| OptionQuote::new(k, tau, price, OptionType::Call))
    .collect()
}

#[test]
fn vg_pricer_matches_reference() {
  // Verify our internal Fourier pricer reproduces reference Vg prices
  let params = [0.2, -0.1, 0.5]; // sigma, theta, nu
  for (i, &k) in STRIKES.iter().enumerate() {
    let price = fourier_call_price(
      LevyModelType::VarianceGamma,
      &params,
      100.0,
      k,
      0.05,
      0.0,
      1.0,
    );
    assert!(
      (price - VG_REF[i]).abs() < 0.05,
      "Vg K={k}: got {price:.6}, expected {:.6}",
      VG_REF[i]
    );
  }
}

#[test]
fn mjd_pricer_matches_reference() {
  // Verify our internal Fourier pricer reproduces reference MJD prices
  let params = [0.15, 1.0, -0.05, 0.1]; // sigma, lambda, mu_j, sigma_j
  for (i, &k) in STRIKES.iter().enumerate() {
    let price = fourier_call_price(LevyModelType::MertonJd, &params, 100.0, k, 0.05, 0.0, 1.0);
    assert!(
      (price - MJD_REF[i]).abs() < 0.05,
      "MJD K={k}: got {price:.6}, expected {:.6}",
      MJD_REF[i]
    );
  }
}

#[test]
fn vg_calibrate_recovers_reference_prices() {
  let calibrator = LevyCalibrator::new(
    LevyModelType::VarianceGamma,
    100.0,
    0.05,
    0.0,
    calls(&STRIKES, &VG_REF, 1.0),
  );
  let result = calibrator.calibrate(None).unwrap();
  assert!(
    result.loss.get(LossMetric::Rmse) < 0.1,
    "Vg RMSE={:.6}",
    result.loss.get(LossMetric::Rmse)
  );
}

#[test]
fn mjd_calibrate_recovers_reference_prices() {
  let calibrator = LevyCalibrator::new(
    LevyModelType::MertonJd,
    100.0,
    0.05,
    0.0,
    calls(&STRIKES, &MJD_REF, 1.0),
  );
  let result = calibrator.calibrate(None).unwrap();
  assert!(
    result.loss.get(LossMetric::Rmse) < 0.1,
    "MJD RMSE={:.6}",
    result.loss.get(LossMetric::Rmse)
  );
}

#[test]
fn test_levy_vg_calibrate() {
  let quotes = calls(
    &[90.0, 95.0, 100.0, 105.0, 110.0],
    &[12.5, 9.0, 6.2, 4.0, 2.3],
    0.5,
  );
  let calibrator = LevyCalibrator::new(LevyModelType::VarianceGamma, 100.0, 0.03, 0.01, quotes);
  let result = calibrator.calibrate(None).unwrap();
  assert!(result.rmse().is_finite() && result.iterations.is_some());
}

/// `to_model` for `LevyModelType::Nig` builds a `NigFourier` whose `price_call` matches the
/// calibrator's own `fourier_call_price`.
#[test]
fn nig_to_model_matches_calibrator_pricer() {
  // Pick a representative NIG triple from the literature (Schoutens 2003 §5.3
  // SPX-style: α=10, β=-5, δ=0.5, mildly skewed/heavy-tailed).
  let params = vec![10.0, -5.0, 0.5];
  let s = 100.0;
  let r = 0.05;
  let q = 0.0;
  let t = 1.0;

  let result = Calibrated::new(
    LevyParams {
      values: params.clone(),
      model_type: LevyModelType::Nig,
    },
    CalibrationLossScore::default(),
    true,
  );
  let model = result.to_model();

  for &k in &STRIKES {
    let from_calibrator = fourier_call_price(LevyModelType::Nig, &params, s, k, r, q, t);
    let from_model = <LevyModel as ModelPricer>::price_call(&model, s, k, r, q, t);
    assert!(
      (from_model - from_calibrator).abs() < 1e-6,
      "NIG to_model mismatch at K={k}: model={from_model:.10} calibrator={from_calibrator:.10}",
    );
  }
}

#[test]
fn test_levy_merton_calibrate() {
  let quotes = calls(
    &[90.0, 95.0, 100.0, 105.0, 110.0],
    &[12.5, 9.0, 6.2, 4.0, 2.3],
    0.5,
  );
  let calibrator = LevyCalibrator::new(LevyModelType::MertonJd, 100.0, 0.03, 0.01, quotes);
  let result = calibrator.calibrate(None).unwrap();
  assert!(result.rmse().is_finite() && result.params.values.len() == 4);
}

/// The trail opens at the clamped start and each later entry is an accepted step, so the
/// objective falls along it to the fit.
#[test]
fn the_trail_opens_at_the_start_and_records_market_minus_model() {
  let quotes = calls(&STRIKES, &VG_REF, 1.0);
  let start = LevyParams {
    values: vec![0.25, -0.05, 0.4],
    model_type: LevyModelType::VarianceGamma,
  };
  let fit = LevyCalibrator::new(
    LevyModelType::VarianceGamma,
    100.0,
    0.05,
    0.0,
    quotes.clone(),
  )
  .with_start(start.clone())
  .with_options(CalibrationOptions::default().with_record_history(true))
  .calibrate(None)
  .unwrap();
  assert!(fit.history.len() > 1, "{} entries", fit.history.len());
  assert_eq!(fit.history[0].params, start);
  assert_eq!(
    fit.history.last().map(|entry| &entry.params),
    Some(&fit.params)
  );
  let objective = |entry: &CalibrationHistory<LevyParams>| entry.residuals.dot(&entry.residuals);
  assert!(
    fit
      .history
      .windows(2)
      .all(|pair| objective(&pair[1]) < objective(&pair[0]))
  );
  for entry in &fit.history {
    let values = &entry.params.values;
    for (quote, residual) in quotes.iter().zip(&entry.residuals) {
      let (k, tau, side) = (quote.strike, quote.tau, quote.option_type);
      let model = fourier_option_price(
        LevyModelType::VarianceGamma,
        values,
        100.0,
        k,
        0.05,
        0.0,
        tau,
        side,
      );
      assert_eq!(*residual, quote.price - model);
    }
  }
}

#[test]
fn calls_and_puts_fit_together() {
  let truth = VarianceGammaFourier {
    sigma: 0.12,
    theta: -0.14,
    nu: 0.2,
  };
  let quotes = [
    (90.0, OptionType::Put),
    (95.0, OptionType::Put),
    (100.0, OptionType::Call),
  ]
  .into_iter()
  .chain([(105.0, OptionType::Call), (110.0, OptionType::Call)])
  .map(|(k, side)| {
    let price = truth.price_option(100.0, k, 0.05, 0.0, 0.5, side);
    OptionQuote::new(k, 0.5, price, side)
  })
  .collect();
  let fit = LevyCalibrator::new(LevyModelType::VarianceGamma, 100.0, 0.05, 0.0, quotes)
    .calibrate(None)
    .unwrap();
  assert!(fit.rmse() < 1e-4, "rmse {}", fit.rmse());
}

#[test]
fn a_start_of_another_model_or_with_a_nan_is_invalid_and_one_of_another_count_mismatched() {
  let calibrator = LevyCalibrator::new(
    LevyModelType::MertonJd,
    100.0,
    0.05,
    0.0,
    calls(&STRIKES, &MJD_REF, 1.0),
  );
  let start = |values: Vec<f64>, model_type| Some(LevyParams { values, model_type });
  for (values, model_type, reason) in [
    (
      vec![0.2, 0.3, -0.05, 0.15],
      LevyModelType::Kou,
      "start.model_type must satisfy",
    ),
    (
      vec![0.2, f64::NAN, -0.05, 0.15],
      LevyModelType::MertonJd,
      "start.values[1] must satisfy",
    ),
  ] {
    match calibrator.calibrate(start(values, model_type)) {
      Err(QuantError::InvalidInput {
        what: "start",
        reason: got,
      }) => {
        assert!(got.starts_with(reason), "{got}")
      }
      other => panic!("{other:?}"),
    }
  }
  assert_eq!(
    calibrator
      .calibrate(start(vec![0.2, 0.3], LevyModelType::MertonJd))
      .unwrap_err(),
    QuantError::DimensionMismatch {
      what: "start values",
      expected: 4,
      got: 2,
    }
  );
}

#[test]
#[should_panic(
  expected = "start.values must satisfy `start.values.len() == 5`, got start.values.len() = 1"
)]
fn with_start_panics_on_a_wrong_count() {
  let _ =
    LevyCalibrator::new(LevyModelType::Kou, 100.0, 0.02, 0.0, Vec::new()).with_start(LevyParams {
      values: vec![0.2],
      model_type: LevyModelType::Kou,
    });
}

#[test]
fn a_capped_fit_counts_the_optimiser_iterations() {
  let fit = LevyCalibrator::new(
    LevyModelType::VarianceGamma,
    100.0,
    0.05,
    0.0,
    calls(&STRIKES, &VG_REF, 1.0),
  )
  .with_options(CalibrationOptions::default().with_max_iterations(1))
  .calibrate(None)
  .unwrap();
  assert_eq!(fit.iterations, Some(1));
  assert_eq!(fit.message.as_deref(), Some("evaluation budget exhausted"));
}

/// Variance Gamma's `(σ, θ, ν)` box clamps, so no coordinate reflects.
#[test]
fn the_jacobian_is_the_slope_of_the_residuals_across_the_box() {
  use super::calibrator::LevyProblem;
  use crate::calibration::least_squares::tests::box_points;
  use crate::calibration::least_squares::tests::jacobian_error;
  let quotes = calls(&STRIKES, &VG_REF, 1.0);
  let calibrator = LevyCalibrator::new(LevyModelType::VarianceGamma, 100.0, 0.05, 0.0, quotes);
  let mut problem = LevyProblem::new(&calibrator, vec![0.2, -0.1, 0.5]);
  let bounds = [(0.01, 2.0), (-1.0, 1.0), (0.01, 5.0)];
  for base in [
    ndarray::array![0.2, -0.1, 0.5],
    ndarray::array![1.2, 0.6, 3.0],
  ] {
    for (label, point) in box_points(&base, &bounds, &[false; 3]) {
      let error = jacobian_error(&mut problem, &point, |_, x| 2e-3 * x.abs().max(1e-2));
      assert!(error < 2e-2, "{label} of {base}: {error:e}");
    }
  }
}
