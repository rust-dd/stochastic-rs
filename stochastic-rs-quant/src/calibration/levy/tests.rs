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
use crate::pricing::cf_quadrature::Walk;
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

// MJD: sigma=0.15, lambda=1.0, muj=-0.05, sigj=0.1
const MJD_REF: [f64; 9] = [
  24.537096, 20.322713, 16.420092, 12.915553, 9.877019, 7.340385, 5.303578, 3.729825, 2.557790,
];

/// The fit's start from `values`: clamped, its coupled parameters read in their own coordinates.
fn project(model: LevyModelType, values: &[f64]) -> Vec<f64> {
  use super::coordinates::to_coordinates;
  use super::coordinates::to_values;
  to_values(model, &to_coordinates(model, values))
}

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
      &mut Walk::Adaptive,
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
    let price = fourier_call_price(
      LevyModelType::MertonJd,
      &params,
      100.0,
      k,
      0.05,
      0.0,
      1.0,
      &mut Walk::Adaptive,
    );
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
    let from_calibrator = fourier_call_price(
      LevyModelType::Nig,
      &params,
      s,
      k,
      r,
      q,
      t,
      &mut Walk::Adaptive,
    );
    let from_model = <LevyModel as ModelPricer>::price_call(&model, s, k, r, q, t);
    assert!(
      (from_model - from_calibrator).abs() < 1e-6,
      "NIG to_model mismatch at K={k}: model={from_model:.10} calibrator={from_calibrator:.10}",
    );
  }
}

/// A fit prices its quotes as its model does: the loss is the blanket `ModelPricer`'s Lewis.
#[test]
fn a_fitted_point_prices_as_its_model_bit_for_bit() {
  for (model_type, reference) in [
    (LevyModelType::VarianceGamma, VG_REF),
    (LevyModelType::MertonJd, MJD_REF),
  ] {
    let quotes = calls(&STRIKES, &reference, 1.0);
    let fit = LevyCalibrator::new(model_type, 100.0, 0.05, 0.0, quotes.clone())
      .calibrate(None)
      .unwrap();
    let model = fit.to_model();
    for quote in &quotes {
      let (k, tau) = (quote.strike, quote.tau);
      let values = &fit.params.values;
      let loss = fourier_call_price(
        model_type,
        values,
        100.0,
        k,
        0.05,
        0.0,
        tau,
        &mut Walk::Adaptive,
      );
      let priced = model.price_call(100.0, k, 0.05, 0.0, tau);
      assert_eq!(
        loss.to_bits(),
        priced.to_bits(),
        "{model_type:?} K = {k}: {loss} vs {priced}"
      );
    }
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
        &mut Walk::Adaptive,
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
  use crate::calibration::least_squares::checks::box_points;
  use crate::calibration::least_squares::checks::jacobian_error;
  let quotes = calls(&STRIKES, &VG_REF, 1.0);
  let calibrator = LevyCalibrator::new(LevyModelType::VarianceGamma, 100.0, 0.05, 0.0, quotes);
  let mut problem = LevyProblem::new(&calibrator, vec![0.2, -0.1, 0.5]);
  let bounds = [(0.01, 2.0), (-1.0, 1.0), (0.01, 5.0)];
  // Every box point keeps `1 − θν − σ²ν/2 > 0`, where Variance Gamma has its `E[e^X]`.
  for base in [
    ndarray::array![0.2, -0.1, 0.5],
    ndarray::array![1.2, -0.6, 0.5],
  ] {
    for (label, point) in box_points(base.view(), &bounds, &[false; 3]) {
      let error = jacobian_error(&mut problem, point.view(), |_, x| 2e-3 * x.abs().max(1e-2));
      assert!(error < 2e-2, "{label} of {base}: {error:e}");
    }
  }
}

/// Without `E[e^X]` (`1 − θν − σ²ν/2 < 0`) the fit's price is NaN, as its model's is.
#[test]
fn a_point_without_a_martingale_prices_nan_as_its_model_does() {
  let values = [1.2, 0.6, 3.0];
  let walk = &mut Walk::Adaptive;
  let price = fourier_call_price(
    LevyModelType::VarianceGamma,
    &values,
    100.0,
    100.0,
    0.05,
    0.0,
    1.0,
    walk,
  );
  let model = VarianceGammaFourier {
    sigma: values[0],
    theta: values[1],
    nu: values[2],
  };
  assert!(price.is_nan() && model.price_call(100.0, 100.0, 0.05, 0.0, 1.0).is_nan());
}

/// On the rule each price's own walk took, `J(h)` is `J(2h)` at every model's default start.
#[test]
fn the_jacobian_at_twice_its_step_is_the_same() {
  use super::calibrator::LevyProblem;
  use super::loss::default_params;
  use crate::calibration::least_squares::checks::step_doubling_error;
  for model in [
    LevyModelType::VarianceGamma,
    LevyModelType::Nig,
    LevyModelType::Cgmy,
    LevyModelType::MertonJd,
    LevyModelType::Kou,
  ] {
    let calibrator = LevyCalibrator::new(model, 100.0, 0.05, 0.0, calls(&STRIKES, &VG_REF, 1.0));
    let problem = LevyProblem::new(&calibrator, default_params(model));
    let error = step_doubling_error(|scale| problem.difference_jacobian(scale));
    assert!(error < 1e-4, "{model:?}: {error:e}");
  }
}

/// `(model, a point of its box without E[e^X], the model's own condition)`: Madan, Carr & Chang's
/// `1 − θν − σ²ν/2`, Barndorff-Nielsen's `α − |β + 1|`, `M − 1` and `η₁ − 1`, each to be positive.
fn points_without_a_martingale() -> Vec<(LevyModelType, Vec<f64>, fn(&[f64]) -> f64)> {
  let vg = |p: &[f64]| 1.0 - p[1] * p[2] - 0.5 * p[0] * p[0] * p[2];
  let nig = |p: &[f64]| p[0] - (p[1] + 1.0).abs().max(p[1].abs());
  vec![
    (LevyModelType::VarianceGamma, vec![1.2, 0.6, 3.0], vg),
    (LevyModelType::VarianceGamma, vec![2.0, -1.0, 5.0], vg),
    (LevyModelType::Nig, vec![3.0, 2.5, 0.5], nig),
    (LevyModelType::Nig, vec![50.0, 49.5, 0.5], nig),
    (LevyModelType::Nig, vec![3.0, -3.0, 0.5], nig),
    (LevyModelType::Cgmy, vec![1.0, 5.0, 0.5, 0.5], |p| {
      p[2] - 1.0
    }),
    (LevyModelType::Cgmy, vec![1.0, 5.0, 1.0, -0.5], |p| {
      p[2] - 1.0
    }),
    (LevyModelType::Kou, vec![0.15, 3.0, 0.5, 1.0, 10.0], |p| {
      p[3] - 1.0
    }),
  ]
}

/// The fit's projection moves each such point 0.01 inside its model's condition, where the price
/// is finite; the point itself prices NaN or without a finite compensator.
#[test]
fn a_projected_point_has_a_martingale_in_every_model() {
  for (model, point, condition) in points_without_a_martingale() {
    let price =
      |p: &[f64]| fourier_call_price(model, p, 100.0, 100.0, 0.05, 0.0, 1.0, &mut Walk::Adaptive);
    assert!(condition(&point) <= 0.0, "{model:?} {point:?}");
    let projected = project(model, &point);
    let margin = condition(&projected);
    assert!(margin >= 0.01 - 1e-12, "{model:?} {projected:?}: {margin}");
    let (inside, outside) = (price(&projected), price(&point));
    assert!(
      inside.is_finite() && inside > 0.0 && inside < 100.0 + 1e-9,
      "{model:?} {projected:?}: {inside}, the point itself {outside}"
    );
    let again = project(model, &projected);
    assert_eq!(again, projected);
    let boxed = projected.iter().zip(super::loss::param_bounds(model));
    assert!(
      boxed.clone().all(|(p, (lo, hi))| lo <= *p && *p <= hi),
      "{model:?} {projected:?}"
    );
  }
}

/// A point inside its condition and its box by more than the margin, NIG's `α` over 50 included,
/// is left as it is, to the map's rounding.
#[test]
fn a_point_with_a_martingale_is_its_own_projection() {
  use super::loss::default_params;
  let inside = [
    (LevyModelType::VarianceGamma, vec![1.2, -0.6, 4.95]),
    (LevyModelType::Nig, vec![6.02, 5.0, 0.5]),
    (LevyModelType::Nig, vec![60.0, 20.0, 0.5]),
    (LevyModelType::Nig, vec![54.0, -45.0, 0.5]),
    (LevyModelType::Cgmy, vec![1.0, 5.0, 1.02, 0.5]),
    (LevyModelType::Kou, vec![0.15, 3.0, 0.5, 1.02, 10.0]),
  ];
  let defaults = [
    LevyModelType::VarianceGamma,
    LevyModelType::Nig,
    LevyModelType::Cgmy,
    LevyModelType::MertonJd,
    LevyModelType::Kou,
  ]
  .map(|model| (model, default_params(model)));
  for (model, point) in inside.into_iter().chain(defaults) {
    let moved = project(model, &point).into_iter().zip(&point);
    let moved = moved.map(|(to, from)| (to - from).abs() / from.abs());
    let moved = moved.fold(0.0, f64::max);
    assert!(moved < 4.0 * f64::EPSILON, "{model:?}: {moved:e}");
  }
}

/// A start without `E[e^X]` has no price to fit from, and its projection is another model's start.
#[test]
fn a_start_without_a_martingale_is_an_error_naming_its_condition() {
  let conditions = [
    (
      LevyModelType::VarianceGamma,
      "`1.0 - theta * nu - 0.5 * sigma * sigma * nu > 0.0`",
    ),
    (
      LevyModelType::Nig,
      "`alpha > beta.abs() && alpha >= (beta + 1.0).abs()`",
    ),
    (LevyModelType::Cgmy, "`m > 1.0`"),
    (LevyModelType::Kou, "`eta1 > 1.0`"),
  ];
  for (model, point, _) in points_without_a_martingale() {
    let start = LevyParams {
      values: point.clone(),
      model_type: model,
    };
    let fit = LevyCalibrator::new(model, 100.0, 0.05, 0.0, calls(&STRIKES, &VG_REF, 1.0))
      .calibrate(Some(start));
    let condition = conditions
      .iter()
      .find(|(named, _)| *named == model)
      .unwrap()
      .1;
    match fit {
      Err(QuantError::InvalidInput {
        what: "start",
        reason,
      }) => assert!(reason.contains(condition), "{model:?} {point:?}: {reason}"),
      other => panic!("{model:?} {point:?}: {other:?}"),
    }
  }
}

#[test]
#[should_panic(expected = "start.values[3] must satisfy `eta1 > 1.0`, got start.values[3] = 1.0")]
fn with_start_panics_on_a_start_without_a_martingale() {
  let _ =
    LevyCalibrator::new(LevyModelType::Kou, 100.0, 0.02, 0.0, Vec::new()).with_start(LevyParams {
      values: vec![0.15, 3.0, 0.5, 1.0, 10.0],
      model_type: LevyModelType::Kou,
    });
}

/// From the domain's edge, the projection of each such point, every accepted step stays inside.
#[test]
fn a_fit_started_on_the_domains_edge_runs_inside_it() {
  for (model, mut point, condition) in points_without_a_martingale() {
    point = project(model, &point);
    let options = CalibrationOptions::default()
      .with_max_iterations(12)
      .with_record_history(true);
    let fit = LevyCalibrator::new(model, 100.0, 0.05, 0.0, calls(&STRIKES, &VG_REF, 1.0))
      .with_start(LevyParams {
        values: point.clone(),
        model_type: model,
      })
      .with_options(options)
      .calibrate(None)
      .unwrap_or_else(|error| panic!("{model:?} {point:?}: {error}"));
    assert!(fit.rmse().is_finite(), "{model:?} {point:?}: {fit:?}");
    for entry in &fit.history {
      let margin = condition(&entry.params.values);
      assert!(margin >= 0.01 - 1e-12, "{model:?} {:?}", entry.params);
    }
    let first = fit.history[0].residuals.dot(&fit.history[0].residuals);
    let last = fit.history.last().unwrap();
    assert!(last.residuals.dot(&last.residuals) <= first);
  }
}
