use ndarray::Array1;

use super::loss::RoughHestonProblem;
use super::params::bounds;
use super::params::folded;
use super::params::held;
use super::params::room_in;
use super::*;
use crate::LossMetric;
use crate::OptionQuote;
use crate::OptionType;
use crate::QuantError;
use crate::calibration::CalibrationOptions;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::calibration::least_squares::checks::assert_slope;
use crate::calibration::least_squares::checks::beside_points;
use crate::calibration::least_squares::checks::beside_reference;
use crate::calibration::least_squares::checks::box_points;
use crate::calibration::least_squares::checks::natural_jacobian_by_differences;
use crate::calibration::least_squares::checks::step_doubling_error;
use crate::calibration::least_squares::checks::worst_relative_error;
use crate::traits::Calibrator;
use crate::traits::ModelPricer;
use crate::traits::ToModel;

const S: f64 = 100.0;
const R: f64 = 0.02;

const TRUTH: RoughHestonParams = RoughHestonParams {
  v0: 0.04,
  kappa: 1.0,
  theta: 0.05,
  nu: 0.3,
  rho: -0.65,
  hurst: 0.1,
};

const START: RoughHestonParams = RoughHestonParams {
  v0: 0.05,
  kappa: 1.4,
  theta: 0.04,
  nu: 0.2,
  rho: -0.5,
  hurst: 0.15,
};

/// Calls of `model` at each maturity's strikes.
fn quotes(model: &RoughHeston, taus: &[f64], strikes: &[f64]) -> Vec<OptionQuote> {
  let pairs = taus
    .iter()
    .flat_map(|&tau| strikes.iter().map(move |&k| (k, tau)));
  pairs
    .map(|(k, tau)| {
      OptionQuote::new(
        k,
        tau,
        model.price_call(S, k, R, 0.0, tau),
        OptionType::Call,
      )
    })
    .collect()
}

/// The box of the selectors whose `ν` runs to 2.
const BOX: [(f64, f64); 6] = [
  (0.005, 0.25),
  (0.01, 10.0),
  (0.001, 1.0),
  (0.01, 2.0),
  (-0.9999, 0.9999),
  (0.005, 0.495),
];

/// A scheme coarse enough for a stencil sweep: the fit's mechanics do not read the selector.
const COARSE: RoughHestonCf = RoughHestonCf::Adams { steps: 8 };

fn coarse_calibrator() -> RoughHestonCalibrator {
  let model = RoughHeston::new(TRUTH, COARSE);
  RoughHestonCalibrator::new(S, R, 0.0, quotes(&model, &[0.5], &[90.0, 100.0, 110.0]))
    .with_cf(COARSE)
}

#[test]
fn a_start_or_a_selector_off_its_domain_is_an_error_and_a_panic_in_its_builder() {
  let calibrator = coarse_calibrator();
  let nan = RoughHestonParams {
    hurst: f64::NAN,
    ..START
  };
  let start = calibrator.calibrate(Some(nan));
  assert!(
    matches!(start, Err(QuantError::InvalidInput { what: "start", .. })),
    "{start:?}"
  );
  let mut written = calibrator.clone();
  written.cf = RoughHestonCf::Pade { order: 7 };
  let cf = written.calibrate(None);
  assert!(
    matches!(cf, Err(QuantError::InvalidInput { what: "cf", .. })),
    "{cf:?}"
  );
  let with_start = std::panic::catch_unwind(|| coarse_calibrator().with_start(nan));
  let with_cf =
    std::panic::catch_unwind(|| coarse_calibrator().with_cf(RoughHestonCf::Adams { steps: 0 }));
  assert!(with_start.is_err() && with_cf.is_err());
  let mut below = calibrator.clone();
  below.quotes[1].price = 1e-9;
  let weight = below.calibrate(None);
  assert!(
    matches!(weight, Err(QuantError::InvalidQuote { index: 1, .. })),
    "{weight:?}"
  );
  assert_eq!(
    (calibrator.s(), calibrator.r(), calibrator.q()),
    (S, R, 0.0)
  );
  assert_eq!(
    RoughHestonCalibrator::new(S, R, 0.0, Vec::new()).cf,
    RoughHestonCf::default()
  );
}

/// A coordinate's room is its folded value's distance to each bound, swapped where the raw point
/// lies reflected; `H` folds at 0.005 and 0.495, inside `(0, ½)`.
#[test]
fn the_room_of_each_coordinate_is_its_distance_to_its_bounds() {
  assert_eq!(bounds(&COARSE), BOX);
  let base = Array1::from(START);
  let rooms = |j: usize, value: f64| {
    let mut x = base.clone();
    x[j] = value;
    let room = room_in(BOX)(x.view());
    (room.down[j], room.up[j], folded(x.view(), &BOX))
  };
  for (j, value, want) in [
    (5, 0.1, (0.095, 0.395)),
    (5, 0.494, (0.489, 0.001)),
    (5, 0.496, (0.001, 0.489)),
    (5, 0.004, (0.489, 0.001)),
    (4, -1.0001, (1.9996, 2e-4)),
    (1, 0.02, (0.01, 9.98)),
    (3, 2.5, (0.5, 1.49)),
  ] {
    let (down, up, point) = rooms(j, value);
    assert!(
      (down - want.0).abs() < 1e-12 && (up - want.1).abs() < 1e-12,
      "{j} at {value}: {down} {up}"
    );
    assert_eq!(point.violation(), None, "{j} at {value}: {point:?}");
  }
}

/// A pull acts on `(V₀, κ, θ, ν, ρ, H)` themselves: the folds' slopes, −1 where reflected.
#[test]
fn the_natural_jacobian_matches_central_differences() {
  let calibrator = coarse_calibrator();
  let mut problem = RoughHestonProblem::new(&calibrator, START, bounds(&calibrator.cf)).unwrap();
  for x in [
    ndarray::array![0.04, 1.0, 0.05, 0.3, -0.65, 0.1],
    ndarray::array![0.3, 10.5, 0.05, 2.2, -1.2, 0.6],
  ] {
    problem.set_params(x.view());
    let analytic = problem.natural_jacobian();
    let numeric = natural_jacobian_by_differences(&mut problem, x.view(), 1e-6);
    let worst = worst_relative_error(analytic.view(), numeric.view());
    assert!(worst < 1e-6, "{x}: {worst:e}");
  }
}

/// Each column within 2e-2 of `‖D‖` on the box and beside a bound, or within what its residuals'
/// rounding explains.
#[test]
fn the_jacobian_is_the_slope_of_the_residuals_across_the_box() {
  let calibrator = coarse_calibrator();
  let mut problem = RoughHestonProblem::new(&calibrator, START, bounds(&calibrator.cf)).unwrap();
  let on_the_box = |_: usize, x: f64| 2e-3 * x.abs().max(1e-2);
  let base = problem.params();
  for (label, point) in box_points(base.view(), &BOX, &[true; 6]) {
    assert_slope(&mut problem, &label, point.view(), on_the_box, 2e-2);
  }
  for (label, point) in beside_points(base.view(), &BOX) {
    assert_slope(&mut problem, &label, point.view(), beside_reference, 2e-2);
  }
}

/// On the rules each price's own walk took, the approximant's Jacobian at twice its step is the
/// same: no stencil straddles a stop of the quadrature.
#[test]
fn the_jacobian_at_twice_its_step_is_the_same() {
  let model = RoughHeston::new(TRUTH, RoughHestonCf::default());
  let market = quotes(&model, &[0.1, 1.0], &[95.0, 105.0]);
  let calibrator = RoughHestonCalibrator::new(S, R, 0.0, market);
  let problem = RoughHestonProblem::new(&calibrator, START, bounds(&calibrator.cf)).unwrap();
  let error = step_doubling_error(|scale| problem.difference_jacobian(scale));
  assert!(error < 1e-4, "{error:e}");
}

/// Quotes of the default selector at `TRUTH`, fitted from `START`: the trail's objective falls,
/// the fit converges onto its quotes, and its model is the selector's at the fitted parameters.
#[test]
fn the_fit_recovers_the_parameters_its_quotes_were_priced_at() {
  let truth = RoughHeston::new(TRUTH, RoughHestonCf::default());
  let market = quotes(&truth, &[0.05, 0.25, 1.0], &[90.0, 100.0, 110.0]);
  let options = CalibrationOptions::default().with_record_history(true);
  let calibrator = RoughHestonCalibrator::new(S, R, 0.0, market.clone())
    .with_start(START)
    .with_options(options);
  let fit = calibrator.calibrate(None).unwrap();
  let rmse = fit.loss.get(LossMetric::Rmse);
  assert!(
    fit.converged && rmse < 1e-6,
    "{rmse:e}: {:?} {:?}",
    fit.message,
    fit.params
  );
  let objective = |i: usize| fit.history[i].residuals.dot(&fit.history[i].residuals);
  assert!(fit.history.len() > 2 && objective(fit.history.len() - 1) < 1e-6 * objective(0));
  assert_eq!(fit.history[0].params, START);
  let (got, want) = (Array1::from(fit.params), Array1::from(TRUTH));
  for j in 0..6 {
    let gap = (got[j] - want[j]).abs() / want[j].abs();
    assert!(gap < 1e-6, "coordinate {j}: {} vs {}", got[j], want[j]);
  }
  let model = fit.to_model();
  assert_eq!(
    (model.cf(), model.params()),
    (RoughHestonCf::default(), fit.params)
  );
  for quote in &market {
    let price = model.price_call(S, quote.strike, R, 0.0, quote.tau);
    assert!((price - quote.price).abs() < 1e-5, "{quote:?}: {price}");
  }
}

/// The result converts by the selector the fit priced with, whatever the default is.
#[test]
fn a_fit_under_another_selector_converts_to_that_selector_s_model() {
  let cf = RoughHestonCf::Adams { steps: 16 };
  let truth = RoughHeston::new(TRUTH, cf);
  let market = quotes(&truth, &[0.25, 1.0], &[90.0, 100.0, 110.0]);
  let options = CalibrationOptions::default()
    .with_record_history(true)
    .with_max_iterations(6);
  let fit = RoughHestonCalibrator::new(S, R, 0.0, market)
    .with_cf(cf)
    .with_start(START)
    .with_options(options)
    .calibrate(None)
    .unwrap();
  assert_eq!(fit.diagnostics, cf);
  assert_eq!(fit.to_model().cf(), cf);
  let losses = fit
    .history
    .iter()
    .map(|entry| entry.loss.get(LossMetric::Rmse));
  let losses = losses.collect::<Vec<_>>();
  assert!(
    losses.len() > 1 && losses[losses.len() - 1] < 0.5 * losses[0],
    "{losses:?}"
  );
}

/// The (4,4) approximant fitted to the Adams solution's quotes lands within its own accuracy:
/// each bound is twice the gap measured, an rmse of 1.9e-4 on a spot of 100 the first.
#[test]
fn the_approximant_fits_the_adams_solution_s_quotes_to_its_accuracy() {
  let truth = RoughHeston::new(TRUTH, RoughHestonCf::Adams { steps: 100 });
  let market = quotes(&truth, &[0.05, 0.25, 1.0], &[90.0, 100.0, 110.0]);
  let fit = RoughHestonCalibrator::new(S, R, 0.0, market)
    .with_start(START)
    .calibrate(None)
    .unwrap();
  let rmse = fit.loss.get(LossMetric::Rmse);
  assert!(fit.converged && rmse < 4e-4, "{rmse:e}: {:?}", fit.message);
  let (got, want) = (Array1::from(fit.params), Array1::from(TRUTH));
  for (j, bound) in [4e-4, 3e-2, 6e-3, 7e-3, 4e-3, 3.2e-2]
    .into_iter()
    .enumerate()
  {
    let gap = (got[j] - want[j]).abs() / want[j].abs();
    assert!(gap < bound, "coordinate {j}: {} vs {}", got[j], want[j]);
  }
}

/// The approximant is validated to `ν = 0.4`, so its box ends there; quotes priced at `ν = 1`
/// hold its fit on that bound, and the Adams fit, whose box runs to 2, recovers them.
#[test]
fn a_pade_fit_stops_at_its_vol_of_vol_bound_where_an_adams_fit_goes_on() {
  let cf = RoughHestonCf::Adams { steps: 16 };
  assert_eq!(bounds(&RoughHestonCf::default())[3], (0.01, 0.4));
  assert_eq!(bounds(&RoughHestonCf::Pade { order: 3 })[3], (0.01, 0.4));
  assert_eq!(
    (bounds(&cf)[3], bounds(&RoughHestonCf::LIFTED)[3]),
    ((0.01, 2.0), (0.01, 2.0))
  );
  assert_eq!(bounds(&RoughHestonCf::default())[4], (-0.85, 0.75));
  assert_eq!(bounds(&RoughHestonCf::Pade { order: 3 })[4], (-0.99, 0.85));
  assert_eq!(bounds(&cf)[4], (-0.9999, 0.9999));
  let capped = CalibrationOptions::default().with_max_iterations(40);
  let truth = RoughHestonParams { nu: 1.0, ..TRUTH };
  let market = quotes(
    &RoughHeston::new(truth, cf),
    &[0.25, 1.0],
    &[90.0, 100.0, 110.0],
  );
  let pade = RoughHestonCalibrator::new(S, R, 0.0, market.clone())
    .with_start(RoughHestonParams { nu: 0.399, ..TRUTH })
    .with_options(capped.clone().with_max_iterations(20))
    .calibrate(None)
    .unwrap();
  let stop = pade.message.as_deref();
  assert!(
    !pade.converged && stop == Some("stopped at the bound of nu"),
    "{stop:?}"
  );
  assert_eq!(pade.params.nu, 0.4);
  let adams = RoughHestonCalibrator::new(S, R, 0.0, market)
    .with_cf(cf)
    .with_start(RoughHestonParams { nu: 0.8, ..TRUTH })
    .with_options(capped)
    .calibrate(None)
    .unwrap();

  assert!(
    adams.converged && (adams.params.nu - 1.0).abs() < 1e-6,
    "{:?}",
    adams.params
  );
}

/// A bound of the model's own box is not a validity limit: quotes priced at `ρ = −0.9999` are
/// fitted on that bound, a constrained optimum, and the fit is converged.
#[test]
fn a_fit_on_a_bound_of_the_model_s_box_is_converged() {
  let cf = RoughHestonCf::Adams { steps: 16 };
  let truth = RoughHestonParams {
    rho: -0.9999,
    ..TRUTH
  };
  let market = quotes(
    &RoughHeston::new(truth, cf),
    &[0.25, 1.0],
    &[90.0, 100.0, 110.0],
  );
  let fit = RoughHestonCalibrator::new(S, R, 0.0, market)
    .with_cf(cf)
    .with_start(RoughHestonParams {
      rho: -0.95,
      ..TRUTH
    })
    .with_options(CalibrationOptions::default().with_max_iterations(40))
    .calibrate(None)
    .unwrap();
  assert!(fit.converged, "{:?}: {:?}", fit.message, fit.params);
  assert!((fit.params.rho + 0.9999).abs() < 1e-6, "{:?}", fit.params);
}

/// Only a validity bound inside the model's box is a stop: `ν`'s floor and the solves' bounds are
/// the model's.
#[test]
fn a_validity_bound_is_named_and_a_model_bound_is_not() {
  let pade = RoughHestonCf::default();
  let at = |nu: f64, rho: f64| RoughHestonParams { nu, rho, ..TRUTH };
  assert_eq!(held(at(0.4, -0.65), &pade), Some("nu"));
  assert_eq!(held(at(0.3, -0.85), &pade), Some("rho"));
  assert_eq!(held(at(0.3, 0.75), &pade), Some("rho"));
  assert_eq!(
    held(at(0.3, -0.99), &RoughHestonCf::Pade { order: 3 }),
    Some("rho")
  );
  assert_eq!(held(at(0.01, -0.65), &pade), None);
  assert_eq!(held(at(0.3, -0.65), &pade), None);
  assert_eq!(held(at(2.0, -0.9999), &RoughHestonCf::ADAMS), None);
  assert_eq!(held(at(2.0, 0.9999), &RoughHestonCf::LIFTED), None);
}

/// At the corners of each order's validity box a call is finite and no more than 1e-6 under its
/// intrinsic value: the approximant's own error where the time value is smaller, as Adams shows.
#[test]
fn inside_its_validity_box_the_approximant_prices_every_corner() {
  for order in [3, 4] {
    let cf = RoughHestonCf::Pade { order };
    let b = bounds(&cf);
    for corner in 0..64 {
      let pick = |j: usize| if corner >> j & 1 == 0 { b[j].0 } else { b[j].1 };
      let p = RoughHestonParams::new(pick(0), pick(1), pick(2), pick(3), pick(4), pick(5));
      let model = RoughHeston::new(p, cf);
      for (tau, k) in [(0.05, 0.9), (1.0, 0.9), (0.05, 1.1)] {
        let call = model.price_call(1.0, k, 0.0, 0.0, tau);
        let floor = (1.0 - k).max(0.0) - 1e-6;
        assert!(
          call >= floor && call <= 1.0,
          "{p:?} order {order} at ({tau}, {k}): {call}"
        );
      }
    }
  }
}
