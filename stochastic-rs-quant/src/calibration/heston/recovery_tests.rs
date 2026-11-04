use super::HestonCalibrator;
use super::HestonJacobianMethod;
use super::HestonParams;
use crate::OptionType;
use crate::calibration::levy::MarketSlice;
use crate::pricing::fourier::HestonFourier;
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

fn surface(truth: &HestonParams) -> Vec<MarketSlice> {
  let model = HestonFourier {
    v0: truth.v0,
    kappa: truth.kappa,
    theta: truth.theta,
    sigma: truth.sigma,
    rho: truth.rho,
    r: RATE,
    q: DIVIDEND,
  };
  MATURITIES
    .iter()
    .map(|&t| MarketSlice {
      strikes: STRIKES.to_vec(),
      prices: STRIKES
        .iter()
        .map(|&k| model.price_call(SPOT, k, RATE, DIVIDEND, t))
        .collect(),
      is_call: vec![true; STRIKES.len()],
      t,
    })
    .collect()
}

fn calibrator(truth: &HestonParams, jacobian_method: HestonJacobianMethod) -> HestonCalibrator {
  let mut calibrator = HestonCalibrator::from_slices(
    None,
    &surface(truth),
    SPOT,
    RATE,
    Some(DIVIDEND),
    OptionType::Call,
    false,
  );
  calibrator.set_jacobian_method(jacobian_method);
  calibrator
}

fn assert_recovers(
  calibrator: &HestonCalibrator,
  truth: &HestonParams,
  start: Option<HestonParams>,
) {
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
fn finite_difference_jacobian_recovers_the_same_parameter_sets() {
  let start = HestonParams {
    v0: 0.05,
    kappa: 2.0,
    theta: 0.05,
    sigma: 0.4,
    rho: -0.5,
  };
  for truth in &CASES {
    assert_recovers(
      &calibrator(truth, HestonJacobianMethod::NumericFiniteDiff),
      truth,
      Some(start.clone()),
    );
  }
}

#[test]
fn reported_rmse_is_the_price_error_of_the_fitted_pricer() {
  let truth = &CASES[0];
  let result = calibrator(truth, HestonJacobianMethod::CuiAnalytic)
    .calibrate(Some(truth.clone()))
    .unwrap();
  assert!(
    result.rmse() < 1e-6,
    "rmse {:e} at an exact fit",
    result.rmse()
  );
}

#[test]
fn termination_reason_accompanies_converged() {
  let result = calibrator(&CASES[1], HestonJacobianMethod::CuiAnalytic)
    .calibrate(None)
    .unwrap();
  let successful = ["Converged", "Orthogonal", "ResidualsZero"]
    .iter()
    .any(|reason| result.termination.starts_with(reason));
  assert_eq!(result.message(), Some(result.termination.as_str()));
  assert!(
    result.converged && successful,
    "converged = {}, termination = {}",
    result.converged,
    result.termination
  );
}

#[test]
fn feller_projection_keeps_a_below_feller_fit_on_the_boundary() {
  let mut calibrator = calibrator(&CASES[0], HestonJacobianMethod::CuiAnalytic);
  calibrator.set_feller_projection(true);
  let fit = calibrator.calibrate(None).unwrap().params;
  assert!(
    fit.sigma * fit.sigma <= 2.0 * fit.kappa * fit.theta * (1.0 + 1e-12),
    "{fit:?}"
  );
}

#[test]
fn residual_weights_leave_an_exact_fit_in_place() {
  let weights = MATURITIES
    .iter()
    .flat_map(|_| STRIKES.iter().map(|k| 1.0 + (k / SPOT - 1.0).abs()))
    .collect::<Vec<_>>();
  for truth in &CASES {
    let mut calibrator = calibrator(truth, HestonJacobianMethod::CuiAnalytic);
    calibrator.set_residual_weights(weights.clone()).unwrap();
    assert_recovers(&calibrator, truth, None);
  }
}
