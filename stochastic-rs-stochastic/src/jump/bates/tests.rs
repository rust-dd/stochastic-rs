use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::normal::SimdNormal;

use super::*;
use crate::euler::EulerSpec;
use crate::euler::EulerSystem;

fn make_bates(drift: Drift<f64>) -> Bates<f64> {
  Bates::new(
    drift,
    1.0,
    0.25,
    0.0,
    0.0,
    0.0,
    0.0,
    SimdNormal::new(0.0, 1.0),
    8,
    1.0,
    0.0,
    Some(1.0),
    VarianceBoundary::Floor,
    Unseeded,
  )
}

/// Mean and standard error of a sample.
fn mean_and_se(xs: &[f64]) -> (f64, f64) {
  let n = xs.len() as f64;
  let mean = xs.iter().sum::<f64>() / n;
  let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
  (mean, (var / n).sqrt())
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
      panic!("a floored Bates steps through the Bates family");
    };
    assert_eq!(drift_c, rate - 0.25);
  }
}

#[test]
fn the_lognormal_jump_never_crosses_zero_and_has_its_compensator_as_its_mean() {
  let (nu, omega) = (-0.1_f64, 0.2);
  let law = LogNormalJump::new(nu, omega);
  assert!(((nu + 0.5 * omega * omega).exp() - 1.0 - law.mean()).abs() < 1e-15);
  let mut rng = Deterministic::new(11).rng();
  let draws = (0..200_000)
    .map(|_| law.sample(&mut rng))
    .collect::<Vec<f64>>();
  assert!(draws.iter().all(|&y| y > -1.0));
  let (mean, se) = mean_and_se(&draws);
  assert!(
    (mean - law.mean()).abs() < 5.0 * se,
    "sample mean {mean} against {} (se {se})",
    law.mean()
  );
}

#[test]
fn compensated_by_its_mean_the_lognormal_jump_leaves_the_price_a_martingale() {
  let law = LogNormalJump::new(-0.1_f64, 0.2);
  let bates = Bates::new(
    Drift::Mu(0.0),
    5.0,
    law.mean(),
    0.0,
    0.0,
    0.0,
    0.0,
    law,
    33,
    100.0,
    0.0,
    Some(1.0),
    VarianceBoundary::Floor,
    Deterministic::new(5),
  );
  let terminal = bates
    .sample_par(20_000)
    .iter()
    .map(|[s, _]| s[32])
    .collect::<Vec<f64>>();
  let (mean, se) = mean_and_se(&terminal);
  assert!(
    (mean - 100.0).abs() < 5.0 * se,
    "E[S_T] {mean} against S_0 = 100 (se {se})"
  );
}
