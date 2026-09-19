use ndarray::Array1;

use crate::OptionType;
use crate::calibration::Regularization;
use crate::calibration::heston::HestonCalibrator;
use crate::calibration::heston::HestonParams;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::pricing::heston::HestonPricer;
use crate::traits::Calibrator;

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
  let pricer = HestonPricer::new(
    TRUTH.v0,
    TRUTH.rho,
    TRUTH.kappa,
    TRUTH.theta,
    TRUTH.sigma,
    Some(0.0),
  );
  let prices: Vec<f64> = strikes
    .iter()
    .map(|&k| pricer.call_put(s, k, r, 0.0, tau).0)
    .collect();
  let mut calibrator = HestonCalibrator::new(
    Some(HestonParams {
      v0: 0.05,
      kappa: 1.0,
      theta: 0.05,
      sigma: 0.6,
      rho: -0.3,
    }),
    Array1::from_vec(prices),
    Array1::from_elem(strikes.len(), s),
    Array1::from_vec(strikes.to_vec()),
    r,
    None,
    tau,
    OptionType::Call,
    None,
    None,
    None,
    false,
  );
  calibrator.regularization = regularization;
  calibrator
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

/// The augmented Jacobian carries one extra row per parameter, with a single
/// positive entry on the regularised coordinate.
#[test]
fn augmented_jacobian_has_the_penalty_rows() {
  let reg = Regularization::new(vec![0.0; 5], vec![0.0, 4.0, 0.0, 0.0, 0.0]);
  let problem = calibrator(Some(reg));
  let jacobian = problem.jacobian().expect("jacobian");
  let n = problem.c_market.len();
  assert_eq!(jacobian.nrows(), n + 5);
  for col in 0..5 {
    for row in 0..5 {
      let value = jacobian[(n + row, col)];
      if row == 1 && col == 1 {
        assert!(value > 0.0, "kappa row must carry a positive derivative");
      } else {
        assert_eq!(value, 0.0);
      }
    }
  }
  assert_eq!(problem.residuals().expect("residuals").len(), n + 5);
}
