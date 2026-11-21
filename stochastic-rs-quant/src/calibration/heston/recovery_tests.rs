use ndarray::s;

use super::HestonCalibrator;
use super::HestonJacobianMethod;
use super::lsq::HestonProblem;
use super::transform::to_optimizer_coordinates;
use crate::OptionQuote;
use crate::OptionType;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationOptions;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::pricing::fourier::HestonFourier;
use crate::pricing::heston::HestonParams;
use crate::traits::CalibrationResult;
use crate::traits::Calibrator;
use crate::traits::ModelPricer;

const SPOT: f64 = 100.0;
const RATE: f64 = 0.02;
const DIVIDEND: f64 = 0.0;
const MATURITIES: [f64; 5] = [0.25, 0.5, 1.0, 2.0, 3.0];
const STRIKES: [f64; 9] = [70.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 130.0];
const TOLERANCE: f64 = 1e-3;

// Feller ratios 2κθ/σ² of 0.48, 3.3, 1.1 and 0.63: a fit must not be pulled onto 2κθ = σ².
const CASES: [HestonParams; 4] = [
  HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.5,
    rho: -0.7,
  },
  HestonParams {
    v0: 0.04,
    kappa: 3.0,
    theta: 0.05,
    sigma: 0.3,
    rho: -0.5,
  },
  HestonParams {
    v0: 0.0225,
    kappa: 1.0,
    theta: 0.0225,
    sigma: 0.2,
    rho: -0.3,
  },
  HestonParams {
    v0: 0.09,
    kappa: 5.0,
    theta: 0.04,
    sigma: 0.8,
    rho: -0.8,
  },
];

// Beside the fallback start, which sits past the cap: one start near the boundary (2κθ/σ² = 1.25),
// one well inside (2.25).
const NEAR_START: HestonParams = HestonParams {
  v0: 0.05,
  kappa: 2.0,
  theta: 0.05,
  sigma: 0.4,
  rho: -0.5,
};

const INNER_START: HestonParams = HestonParams {
  v0: 0.03,
  kappa: 1.5,
  theta: 0.03,
  sigma: 0.2,
  rho: -0.5,
};

const STARTS: [Option<HestonParams>; 3] = [None, Some(NEAR_START), Some(INNER_START)];

fn surface(truth: &HestonParams) -> Vec<OptionQuote> {
  let model = HestonFourier::new(*truth);
  MATURITIES
    .iter()
    .flat_map(|&t| STRIKES.iter().map(move |&k| (k, t)))
    .map(|(k, t)| {
      let price = model.price_call(SPOT, k, RATE, DIVIDEND, t);
      OptionQuote::new(k, t, price, OptionType::Call)
    })
    .collect()
}

fn calibrator(truth: &HestonParams, jacobian_method: HestonJacobianMethod) -> HestonCalibrator {
  HestonCalibrator::new(SPOT, RATE, DIVIDEND, surface(truth)).with_jacobian_method(jacobian_method)
}

fn assert_recovers(
  calibrator: &HestonCalibrator,
  truth: &HestonParams,
  start: Option<HestonParams>,
) -> Calibrated<HestonParams> {
  let result = calibrator.calibrate(start).unwrap();
  let fit = &result.params;
  let errors = [
    fit.v0 - truth.v0,
    fit.kappa - truth.kappa,
    fit.theta - truth.theta,
    fit.sigma - truth.sigma,
    fit.rho - truth.rho,
  ];
  assert!(
    result.converged && errors.iter().all(|error| error.abs() < TOLERANCE),
    "truth {truth:?}: fit {fit:?}, converged = {}",
    result.converged
  );
  result
}

#[test]
fn default_start_recovers_parameters_on_both_sides_of_the_feller_boundary() {
  for truth in &CASES {
    assert_recovers(
      &calibrator(truth, HestonJacobianMethod::CuiAnalytic),
      truth,
      None,
    );
  }
}

#[test]
fn finite_difference_jacobian_recovers_a_surface_on_each_side_of_the_feller_boundary() {
  for truth in [&CASES[0], &CASES[1]] {
    assert_recovers(
      &calibrator(truth, HestonJacobianMethod::NumericFiniteDiff),
      truth,
      Some(NEAR_START),
    );
  }
}

// The fitted Cui pricer sits 3.6e-8 from this Gil–Pelaez surface. The residuals are re-evaluated
// through the coordinate round trip, which need not return the fit's bits: 1.7e-16 apart in rmse.
#[test]
fn reported_rmse_is_the_price_error_of_the_fitted_pricer() {
  let truth = &CASES[2];
  let calibrator =
    calibrator(truth, HestonJacobianMethod::CuiAnalytic).with_feller_projection(true);
  let result = calibrator.calibrate(Some(*truth)).unwrap();
  let mut objective = HestonProblem::new(&calibrator, result.params);
  objective.feller_row_weight = Some(1.0);
  objective.set_params(to_optimizer_coordinates(&result.params).view());
  let residuals = objective.residuals().unwrap();
  let quotes = calibrator.quotes.len();
  assert_eq!(residuals.len(), quotes + 1);
  let prices = residuals.slice(s![..quotes]);
  let price_rmse = (prices.dot(&prices) / quotes as f64).sqrt();
  assert!(
    result.rmse() < 1e-6 && (result.rmse() - price_rmse).abs() <= f64::EPSILON * SPOT,
    "rmse {:e}, the minimised price residuals' {price_rmse:e}",
    result.rmse()
  );
}

#[test]
fn stop_reason_accompanies_converged() {
  let result = calibrator(&CASES[1], HestonJacobianMethod::CuiAnalytic)
    .calibrate(None)
    .unwrap();
  assert!(
    result.converged && result.message() == Some("converged"),
    "{:?}",
    result.message()
  );
  let result = calibrator(&CASES[1], HestonJacobianMethod::CuiAnalytic)
    .with_options(CalibrationOptions::default().with_patience(1))
    .calibrate(None)
    .unwrap();
  assert!(
    !result.converged && result.message() == Some("evaluation budget exhausted"),
    "{:?}",
    result.message()
  );
}

#[test]
fn feller_projection_fits_a_below_feller_surface_on_the_boundary_from_any_start() {
  let fit_from = |start: Option<HestonParams>| {
    let result = calibrator(&CASES[0], HestonJacobianMethod::CuiAnalytic)
      .with_feller_projection(true)
      .with_options(CalibrationOptions::default().with_record_history(true))
      .calibrate(start)
      .unwrap();
    assert!(
      result
        .history
        .iter()
        .all(|step| step.params.satisfies_feller_condition())
    );
    result
  };
  let fallback = fit_from(None);
  let feasible = fit_from(Some(INNER_START));
  for result in [&fallback, &feasible] {
    let fit = &result.params;
    let slack = 1.0 - fit.sigma * fit.sigma / (2.0 * fit.kappa * fit.theta);
    assert!(
      result.converged && fit.satisfies_feller_condition() && slack < 1e-9,
      "{fit:?}"
    );
  }
  assert!(
    (fallback.rmse() - feasible.rmse()).abs() < 1e-9 * fallback.rmse(),
    "rmse {:e} from the fallback start, {:e} from a feasible one",
    fallback.rmse(),
    feasible.rmse()
  );
}

#[test]
fn feller_projection_recovers_a_surface_just_inside_the_boundary_from_every_start() {
  for start in STARTS {
    let calibrator =
      calibrator(&CASES[2], HestonJacobianMethod::CuiAnalytic).with_feller_projection(true);
    let result = assert_recovers(&calibrator, &CASES[2], start);
    assert!(result.rmse() < 1e-6, "rmse {:e}", result.rmse());
  }
}

#[test]
fn residual_weights_leave_an_exact_fit_in_place() {
  let weights = MATURITIES
    .iter()
    .flat_map(|_| STRIKES.iter().map(|k| 1.0 + (k / SPOT - 1.0).abs()))
    .collect::<Vec<_>>();
  for truth in &CASES {
    let calibrator =
      calibrator(truth, HestonJacobianMethod::CuiAnalytic).with_residual_weights(weights.clone());
    assert_recovers(&calibrator, truth, None);
  }
}

/// From a start at the correlation bound the first step throws ρ's logistic coordinate to the
/// other end, where its slope is 1e-6 and no step of the optimiser's leaves.
#[test]
fn a_start_at_the_correlation_bound_recovers_every_truth_set() {
  let start = |rho: f64| Some(HestonParams { rho, ..NEAR_START });
  for (case, rho) in [(0, 1.0), (3, 1.0), (3, 0.99)] {
    let truth = &CASES[case];
    assert_recovers(
      &calibrator(truth, HestonJacobianMethod::CuiAnalytic),
      truth,
      start(rho),
    );
  }
}

/// A 1 % ripple leaves a minimum with an rmse of 0.11551, short of an exact fit's exemption.
#[test]
fn a_start_at_the_correlation_bound_reaches_the_minimum_of_a_rippled_surface() {
  let mut quotes = surface(&CASES[0]);
  for (i, quote) in quotes.iter_mut().enumerate() {
    quote.price *= 1.0 + 0.01 * (i as f64 * 1.7).sin();
  }
  let calibrator = HestonCalibrator::new(SPOT, RATE, DIVIDEND, quotes);
  let start = HestonParams {
    rho: 1.0,
    ..NEAR_START
  };
  for start in [None, Some(start)] {
    let fit = calibrator.calibrate(start).unwrap();
    assert!(
      fit.converged && (fit.rmse() - 0.11551).abs() < 1e-5,
      "{start:?}: {:?}, rmse {}",
      fit.params,
      fit.rmse()
    );
  }
}

#[test]
fn the_restarts_from_a_bound_draw_on_one_iteration_budget() {
  let start = HestonParams {
    rho: 1.0,
    ..NEAR_START
  };
  let calibrator = calibrator(&CASES[0], HestonJacobianMethod::CuiAnalytic);
  let free = calibrator.calibrate(Some(start)).unwrap();
  let cap = free.iterations.unwrap() - 5;
  let capped = calibrator
    .with_options(CalibrationOptions::default().with_max_iterations(cap))
    .calibrate(Some(start))
    .unwrap();
  assert!(!capped.converged && capped.iterations <= Some(cap));
  assert_eq!(capped.message(), Some("evaluation budget exhausted"));
}

/// Capped at the 53 iterations its first pass takes, the fit stops on the cap inside that pass,
/// before it converges and so before a re-seat is asked, and says its budget ran out.
#[test]
fn a_fit_capped_inside_its_first_pass_reports_its_budget() {
  let start = HestonParams {
    rho: 1.0,
    ..NEAR_START
  };
  let capped = calibrator(&CASES[0], HestonJacobianMethod::CuiAnalytic)
    .with_options(CalibrationOptions::default().with_max_iterations(53))
    .calibrate(Some(start))
    .unwrap();
  assert_eq!((capped.converged, capped.iterations), (false, Some(53)));
  assert_eq!(capped.message(), Some("evaluation budget exhausted"));
}

/// The pull holds ρ at its anchor against the quotes, so the price rows alone fall inward there.
#[test]
fn a_fit_held_beside_a_bound_by_its_pull_is_converged() {
  let anchor = HestonParams {
    rho: -0.9995,
    ..CASES[0]
  };
  let pull = crate::calibration::Regularization::new(
    vec![
      anchor.v0,
      anchor.kappa,
      anchor.theta,
      anchor.sigma,
      anchor.rho,
    ],
    vec![0.0, 0.0, 0.0, 0.0, 1e10],
  );
  let fit = calibrator(&CASES[0], HestonJacobianMethod::CuiAnalytic)
    .with_options(CalibrationOptions::default().with_regularization(pull))
    .calibrate(Some(anchor))
    .unwrap();
  assert!(fit.converged, "{:?}: {:?}", fit.message, fit.params);
  assert!(
    (fit.params.rho - anchor.rho).abs() < 1e-4,
    "{}",
    fit.params.rho
  );
}

/// Cui's Jacobian against the residuals' central difference in the optimiser's coordinates: the
/// default start, each truth, a far point, and each coordinate saturated to within 1e-3 of an end.
#[test]
fn the_jacobian_is_the_slope_of_the_residuals_across_the_box() {
  use crate::calibration::least_squares::checks::jacobian_error;
  let calibrator = calibrator(&CASES[0], HestonJacobianMethod::CuiAnalytic);
  let mut problem = HestonProblem::new(&calibrator, NEAR_START);
  let far = ndarray::array![3.0, -3.0, 2.0, -2.0, 4.0];
  let mut points = vec![
    to_optimizer_coordinates(&HestonCalibrator::fallback_params()),
    far,
  ];
  points.extend(CASES.iter().map(to_optimizer_coordinates));
  for coordinate in 0..5 {
    for end in [-6.9, 6.9] {
      let mut point = to_optimizer_coordinates(&NEAR_START);
      point[coordinate] = end;
      points.push(point);
    }
  }
  for point in points {
    let error = jacobian_error(&mut problem, point.view(), |_, _| 1e-5);
    assert!(error < 1e-3, "{point}: {error:e}");
  }
}
