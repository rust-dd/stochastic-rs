use super::*;

#[test]
fn variance_stays_non_negative() {
  let p = BatesSvj::new(
    Drift::Mu(0.05_f64),
    0.5,
    -0.1,
    0.2,
    0.04,
    1.5,
    0.3,
    -0.7,
    256,
    Some(100.0),
    Some(0.04),
    Some(1.0),
    VarianceBoundary::Floor,
    Unseeded,
  );
  let [_s, v] = p.sample();
  assert!(v.iter().all(|x| *x >= 0.0));
}

#[test]
fn price_stays_positive() {
  let p = BatesSvj::new(
    Drift::Mu(0.05_f64),
    0.5,
    -0.1,
    0.2,
    0.04,
    1.5,
    0.3,
    -0.7,
    256,
    Some(100.0),
    Some(0.04),
    Some(1.0),
    VarianceBoundary::Floor,
    Unseeded,
  );
  let [s, _v] = p.sample();
  assert!(s.iter().all(|x| *x > 0.0));
}

#[test]
fn every_drift_form_reaches_the_spot_drift_net_of_the_compensator() {
  use crate::euler::EulerSpec;
  use crate::euler::EulerSystem;
  let forms = [
    (Drift::Mu(0.75), 0.75),
    (Drift::CostOfCarry(0.5), 0.5),
    (Drift::Rates { r: 0.75, r_f: 0.5 }, 0.25),
  ];
  for (drift, rate) in forms {
    let p = BatesSvj::new(
      drift,
      0.5,
      0.0,
      0.0,
      0.04,
      1.5,
      0.3,
      -0.5,
      8,
      Some(1.0),
      Some(0.04),
      Some(1.0),
      VarianceBoundary::Floor,
      Unseeded,
    );
    let EulerSpec::BatesJump { drift_c, .. } = EulerSystem::<f64, 2>::euler_spec(&p) else {
      panic!("a floored BatesSvj steps through the BatesJump family");
    };
    assert_eq!(drift_c, rate, "a zero jump size has a zero compensator");
  }
}
