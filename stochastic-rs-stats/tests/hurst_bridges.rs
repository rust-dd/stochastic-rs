//! `HurstFit::to_fgn` / `to_fbm` build a process at the estimated exponent.

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stats::hurst::HurstEstimator;
use stochastic_rs_stats::hurst::rs::RescaledRange;
use stochastic_rs_stochastic::device::Cpu;
use stochastic_rs_stochastic::noise::fgn::Fgn;
use stochastic_rs_stochastic::traits::ProcessExt;

fn rescaled_range() -> RescaledRange {
  RescaledRange {
    take_differences: false,
    ..RescaledRange::default()
  }
}

#[test]
fn to_fgn_round_trips_through_rescaled_range() {
  let true_h = 0.3;
  let path = Fgn::<f64, _, Cpu>::new(true_h, 4096, Some(1.0), Deterministic::new(42)).sample();
  let result = rescaled_range().estimate(path.view()).unwrap();
  assert!(
    (result.hurst - true_h).abs() < 0.1,
    "H = {:.3}",
    result.hurst
  );

  let rebuilt = result.to_fgn(4096, Some(1.0), Deterministic::new(7));
  assert_eq!(rebuilt.hurst(), result.hurst);
  let reestimated = rescaled_range().estimate(rebuilt.sample().view()).unwrap();
  assert!(
    (reestimated.hurst - true_h).abs() < 0.15,
    "H = {:.3}",
    reestimated.hurst
  );
}

/// `Fbm` samples through `Fgn`, so the round trip above covers the law; this checks the wiring.
#[test]
fn to_fbm_wires_its_arguments_and_samples() {
  let path = Fgn::<f64, _, Cpu>::new(0.6, 2048, Some(1.0), Deterministic::new(3)).sample();
  let result = rescaled_range().estimate(path.view()).unwrap();

  let fbm = result.to_fbm(512, Some(2.0), Deterministic::new(9));
  assert_eq!(fbm.hurst(), result.hurst);
  assert_eq!(fbm.n(), 512);
  assert_eq!(fbm.t(), Some(2.0));
  let sample = fbm.sample();
  assert_eq!(sample.len(), 512);
  assert!(sample.iter().all(|v| v.is_finite()));
}
