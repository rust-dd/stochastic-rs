// docs: stats#hurst-exponent-estimation
//! Backs the Hurst-estimator example on the statistics catalog page.
//!
//! Uses the rescaled-range estimator (`RescaledRange`) rather than the
//! Fukasawa/Whittle one (`stats::hurst::whittle`): Fukasawa estimates the
//! roughness of a *latent volatility* process from a realized-variance
//! series, not the Hurst exponent of a path sampled directly from `Fgn` —
//! feeding path samples straight into it silently recovers nonsense. See
//! `stochastic-rs-stats/src/hurst/whittle.rs`'s own `simulate_log_rv` test
//! helper for the realized-variance construction Fukasawa actually expects.

use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stats::hurst::HurstEstimator;
use stochastic_rs::stats::hurst::rs::RescaledRange;
use stochastic_rs::stochastic::noise::fgn::Fgn;
use stochastic_rs::traits::ProcessExt;

#[test]
fn rescaled_range_recovers_true_h_from_fgn_increments() {
  let true_h = 0.3;
  let path = Fgn::<f64, _>::new(true_h, 4096, Some(1.0), Deterministic::new(42)).sample();

  // `path` is already the stationary fGn increment series, so disable the
  // estimator's default first-differencing (meant for an fBM-like walk).
  let estimator = RescaledRange {
    take_differences: false,
    ..RescaledRange::default()
  };
  let res = estimator.estimate(path.view()).unwrap();
  assert!(
    (res.hurst - true_h).abs() < 0.1,
    "H = {:.3} (true = {:.3})",
    res.hurst,
    true_h
  );
}
