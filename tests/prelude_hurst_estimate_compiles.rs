//! Guards the prelude against a missing stats trait: a consumer importing only
//! `stochastic_rs::prelude::*` plus concrete module paths must still compile a first `.estimate()`.

use stochastic_rs::prelude::*;
use stochastic_rs::simd_rng::Deterministic;

#[test]
fn hurst_estimate_compiles_from_prelude_alone() {
  let path = stochastic_rs::stochastic::noise::fgn::Fgn::<f64, _>::new(
    0.3,
    2048,
    Some(1.0),
    Deterministic::new(11),
  )
  .sample();

  let estimator = stochastic_rs::stats::hurst::rs::RescaledRange {
    take_differences: false,
    ..stochastic_rs::stats::hurst::rs::RescaledRange::default()
  };
  let result = estimator.estimate(path.view()).unwrap();
  assert!(
    (result.hurst - 0.3).abs() < 0.15,
    "H = {:.3} (true = 0.3)",
    result.hurst
  );
}
