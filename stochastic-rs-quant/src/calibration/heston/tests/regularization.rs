use crate::OptionQuote;
use crate::OptionType;
use crate::calibration::CalibrationOptions;
use crate::calibration::Regularization;
use crate::calibration::heston::HestonCalibrator;
use crate::calibration::heston::lsq::HestonProblem;
use crate::calibration::heston::transform::to_optimizer_coordinates;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::calibration::least_squares::checks::natural_jacobian_by_differences;
use crate::calibration::least_squares::checks::worst_relative_error;
use crate::pricing::heston::HestonParams;
use crate::pricing::heston::HestonPricer;
use crate::traits::Calibrator;
use crate::traits::ModelPricer;

const TRUTH: HestonParams = HestonParams {
  v0: 0.04,
  kappa: 2.0,
  theta: 0.04,
  sigma: 0.5,
  rho: -0.5,
};

fn calibrator(regularization: Option<Regularization>) -> HestonCalibrator {
  let (s, r, tau) = (100.0, 0.02, 0.5);
  let strikes = [80.0, 90.0, 100.0, 110.0, 120.0];
  let pricer = HestonPricer::new(TRUTH);
  let quotes = strikes
    .iter()
    .map(|&k| {
      let price = pricer.price_call(s, k, r, 0.0, tau);
      OptionQuote::new(k, tau, price, OptionType::Call)
    })
    .collect();
  let options = CalibrationOptions {
    regularization,
    ..CalibrationOptions::default()
  };
  HestonCalibrator::new(s, r, 0.0, quotes)
    .with_start(HestonParams {
      v0: 0.05,
      kappa: 1.0,
      theta: 0.05,
      sigma: 0.6,
      rho: -0.3,
    })
    .with_options(options)
}

/// A pull on κ alone, heavy against the price residuals, pins κ at the
/// anchor while the other parameters stay free.
#[test]
fn heavy_anchor_on_kappa_pins_it() {
  let anchor = vec![TRUTH.v0, 3.0, TRUTH.theta, TRUTH.sigma, TRUTH.rho];
  let reg = Regularization::new(anchor, vec![0.0, 1e6, 0.0, 0.0, 0.0]);
  let result = calibrator(Some(reg))
    .calibrate(None)
    .expect("calibration runs");
  assert!(
    (result.params.kappa - 3.0).abs() < 1e-2,
    "kappa {}",
    result.params.kappa
  );
}

/// Zero weights take the unregularised path, bit for bit.
#[test]
fn zero_weights_match_the_unregularised_calibration() {
  let plain = calibrator(None).calibrate(None).expect("calibration runs");
  let zero = calibrator(Some(Regularization::uniform(vec![0.1; 5], 0.0)))
    .calibrate(None)
    .expect("calibration runs");
  assert_eq!(plain.params.kappa, zero.params.kappa);
  assert_eq!(plain.params.sigma, zero.params.sigma);
  assert_eq!(plain.params.rho, zero.params.rho);
}

/// Off the Feller cap a pull sees `(v0, κ, θ, σ, ρ)` through the logistic transform's diagonal.
#[test]
fn the_pull_reads_the_physical_parameters_through_the_transform() {
  let calibrator = calibrator(None);
  let start = calibrator.start.unwrap();
  let problem = HestonProblem::new(&calibrator, start);
  let physical = [start.v0, start.kappa, start.theta, start.sigma, start.rho];
  for (natural, value) in problem.natural_params().iter().zip(physical) {
    assert!((natural - value).abs() < 1e-12, "{natural} against {value}");
  }
  for ((row, col), value) in problem.natural_jacobian().indexed_iter() {
    assert_eq!(*value > 0.0, row == col, "({row}, {col}) = {value}");
  }
}

/// Where the cap binds, σ = √(2κθ) moves with κ and θ alone, so its row leaves σ's column.
#[test]
fn under_the_feller_cap_the_sigma_row_moves_onto_kappa_and_theta() {
  let calibrator = calibrator(None).with_feller_projection(true);
  let problem = HestonProblem::new(&calibrator, calibrator.start.unwrap());
  let jacobian = problem.natural_jacobian();
  let sigma_row = jacobian.row(3).to_vec();
  assert!(sigma_row[1] > 0.0 && sigma_row[2] > 0.0, "{sigma_row:?}");
  assert!(sigma_row[0] == 0.0 && sigma_row[3] == 0.0 && sigma_row[4] == 0.0);
  let capped = problem.natural_params()[3];
  assert!(
    (capped - (2.0 * 1.0 * 0.05_f64).sqrt()).abs() < 1e-9,
    "{capped}"
  );
}

#[test]
fn the_natural_jacobian_matches_central_differences_with_the_cap_free_and_binding() {
  let free = HestonParams {
    v0: 0.05,
    kappa: 1.5,
    theta: 0.05,
    sigma: 0.3,
    rho: -0.4,
  };
  let binding = HestonParams {
    kappa: 1.0,
    sigma: 0.6,
    ..free
  };
  for (feller, point) in [
    (false, free),
    (true, free),
    (true, binding),
    (false, binding),
  ] {
    let calibrator = calibrator(None).with_feller_projection(feller);
    let mut problem = HestonProblem::new(&calibrator, point);
    let x = to_optimizer_coordinates(&point);
    problem.set_params(x.view());
    let analytic = problem.natural_jacobian();
    let numeric = natural_jacobian_by_differences(&mut problem, x.view(), 1e-6);
    let worst = worst_relative_error(analytic.view(), numeric.view());
    assert!(worst < 1e-6, "feller {feller}, {point:?}: {worst:e}");
  }
}
