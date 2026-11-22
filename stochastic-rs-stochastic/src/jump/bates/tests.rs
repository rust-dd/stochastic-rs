use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::normal::SimdNormal;

use super::*;
use crate::euler::EulerSpec;
use crate::euler::EulerSystem;
use crate::euler::JumpSizes;

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
    VarianceBoundary::Absorption,
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
      panic!("an absorbing Bates steps through the Bates family");
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
    VarianceBoundary::Absorption,
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

#[test]
fn the_lognormal_law_keeps_the_price_positive_where_a_price_step_crosses_zero() {
  let law = LogNormalJump::new(-1.0_f64, 0.5);
  let quarterly = Bates::new(
    Drift::Mu(0.05),
    1.0,
    law.mean(),
    0.5,
    2.0,
    0.5,
    -0.7,
    law,
    5,
    100.0,
    0.25,
    Some(1.0),
    VarianceBoundary::Absorption,
    Deterministic::new(11),
  );
  let crossed = quarterly
    .sample_par(20_000)
    .iter()
    .filter(|[s, _]| s.iter().any(|&x| x <= 0.0))
    .count();
  assert_eq!(
    crossed, 0,
    "{crossed} of 20000 paths reached a non-positive price"
  );
}

#[test]
fn the_lognormal_law_launches_the_log_price_families_with_the_process_compensator() {
  let law = LogNormalJump::new(-0.1_f64, 0.2);
  for boundary in [VarianceBoundary::Absorption, VarianceBoundary::Reflect] {
    let bates = Bates::new(
      Drift::Mu(0.05),
      5.0,
      0.3,
      0.04,
      1.5,
      0.3,
      -0.6,
      law,
      24,
      100.0,
      0.04,
      Some(1.0),
      boundary,
      Unseeded,
    );
    let drift_c = match (EulerSystem::<f64, 2>::euler_spec(&bates), boundary) {
      (EulerSpec::BatesJump { drift_c, .. }, VarianceBoundary::Absorption)
      | (EulerSpec::BatesJumpReflected { drift_c, .. }, VarianceBoundary::Reflect) => drift_c,
      (spec, _) => panic!("{boundary:?} launched {spec:?}"),
    };
    assert_eq!(drift_c, 0.05 - 5.0 * 0.3);
    assert_eq!(
      EulerSystem::<f64, 2>::jump_sizes(&bates),
      Some(JumpSizes::Normal {
        mean: -0.1,
        sd: 0.2
      })
    );
    assert_eq!(bates.device_fallback(), None);
  }
}
