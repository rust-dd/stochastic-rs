//! Guards `stochastic_rs::prelude::*` against the gap this task found: the
//! umbrella hub re-exported zero stats traits, so a consumer importing only
//! the prelude could not write a first `.estimate()` call at all — a
//! compile failure, not an inconvenience.
//!
//! This is that fresh consumer: nothing here is imported except the
//! prelude (plus concrete, non-trait module paths for the process/
//! estimator types themselves, which bring no methods into scope on their
//! own). Before `HurstEstimator` was added to the prelude, this file did
//! not compile — see `task-6-report.md` for the captured `E0599`.

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
