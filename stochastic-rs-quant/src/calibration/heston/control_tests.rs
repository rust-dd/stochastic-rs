use nalgebra::DVector;

use super::*;
use crate::OptionType;

fn calibrator() -> HestonCalibrator {
  HestonCalibrator::new(
    Some(HestonParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.6,
    }),
    DVector::from_vec(vec![5.0]),
    DVector::from_vec(vec![100.0]),
    DVector::from_vec(vec![100.0]),
    0.03,
    Some(0.01),
    0.25,
    OptionType::Call,
    None,
    None,
    None,
    false,
  )
}

#[test]
fn optimizer_controls_validate_and_persist() {
  let mut value = calibrator();
  value.set_optimizer_tolerance(Some(1e-7)).unwrap();
  value.set_optimizer_patience(20).unwrap();
  assert_eq!(value.optimizer_tolerance, Some(1e-7));
  assert_eq!(value.optimizer_patience, 20);
  value.set_optimizer_tolerance(None).unwrap();
  assert_eq!(value.optimizer_tolerance, None);
}

#[test]
fn invalid_optimizer_controls_fail_closed() {
  let mut value = calibrator();
  assert!(value.set_optimizer_tolerance(Some(0.0)).is_err());
  assert!(value.set_optimizer_tolerance(Some(f64::NAN)).is_err());
  assert!(value.set_optimizer_patience(0).is_err());
}
