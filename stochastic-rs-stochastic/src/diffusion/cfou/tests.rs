use stochastic_rs_core::simd_rng::Unseeded;

use super::Cfou;
use crate::traits::ProcessExt;

#[test]
fn cfou_sample_is_complex_and_finite() {
  let p = Cfou::<f64>::new(
    0.7,
    1.2,
    3.0,
    0.4,
    256,
    Some(0.0),
    Some(0.0),
    Some(1.0),
    Unseeded,
  );
  let z = p.sample();

  assert_eq!(z.len(), 256);
  assert!(z.iter().all(|v| v.re.is_finite() && v.im.is_finite()));
}

#[test]
fn cfou_components_are_finite() {
  let p = Cfou::<f64>::new(
    0.65,
    0.9,
    2.5,
    0.6,
    128,
    Some(0.1),
    Some(-0.1),
    Some(1.0),
    Unseeded,
  );
  let [x1, x2] = p.sample_components();
  assert_eq!(x1.len(), 128);
  assert_eq!(x2.len(), 128);
  assert!(x1.iter().all(|v| v.is_finite()));
  assert!(x2.iter().all(|v| v.is_finite()));
}
