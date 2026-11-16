use stochastic_rs_distributions::normal::SimdNormal;

use super::*;
use crate::euler::EulerSpec;
use crate::euler::EulerSystem;

fn make_bates(drift: Drift<f64>) -> Bates1996<f64, SimdNormal<f64>> {
  Bates1996::new(
    drift,
    1.0,
    0.25,
    0.0,
    0.0,
    0.0,
    0.0,
    SimdNormal::new(0.0, 1.0),
    8,
    Some(1.0),
    Some(0.0),
    Some(1.0),
    VarianceBoundary::Floor,
    Unseeded,
  )
}

#[test]
fn every_drift_form_reaches_the_spot_drift_net_of_the_compensator() {
  let forms = [
    (Drift::Mu(0.75), 0.75),
    (Drift::CostOfCarry(0.5), 0.5),
    (Drift::Rates { r: 0.75, r_f: 0.5 }, 0.25),
  ];
  for (drift, rate) in forms {
    let EulerSpec::Bates1996 { drift_c, .. } =
      EulerSystem::<f64, 2>::euler_spec(&make_bates(drift))
    else {
      panic!("a floored Bates1996 steps through the Bates1996 family");
    };
    assert_eq!(drift_c, rate - 0.25);
  }
}
