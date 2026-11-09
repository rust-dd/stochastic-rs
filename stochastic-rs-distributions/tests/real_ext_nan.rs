//! `max_or_nan` / `min_or_nan` propagate a NaN operand; `Float::max` / `min` drop it.

use stochastic_rs_distributions::RealExt;

#[test]
fn a_nan_operand_makes_the_reduction_nan() {
  assert!(f64::NAN.max_or_nan(1.0).is_nan());
  assert!(1.0_f64.max_or_nan(f64::NAN).is_nan());
  assert!(2.5_f32.min_or_nan(f32::NAN).is_nan());
  assert_eq!(1.0_f64.max_or_nan(2.0), 2.0);
  assert_eq!(1.0_f64.min_or_nan(2.0), 1.0);
  let xs = [1.0_f64, f64::NAN, 3.0];
  assert_eq!(xs.iter().copied().fold(f64::NEG_INFINITY, f64::max), 3.0);
  assert!(
    xs.iter()
      .copied()
      .fold(f64::NEG_INFINITY, f64::max_or_nan)
      .is_nan()
  );
}
