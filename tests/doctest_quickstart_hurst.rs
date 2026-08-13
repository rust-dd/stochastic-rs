// docs: getting-started/quickstart#3-estimate-hurst-from-a-fractional-brownian-path
//! Backs vignette 3 (estimate Hurst) on the quickstart page. Uses the
//! rescaled-range estimator — see `doctest_stats_hurst.rs` for why the
//! Fukasawa/Whittle one is the wrong tool for a raw sampled path.

use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stats::hurst::HurstEstimator;
use stochastic_rs::stats::hurst::rs::RescaledRange;
use stochastic_rs::stochastic::noise::fgn::Fgn;
use stochastic_rs::traits::ProcessExt;

#[test]
fn estimate_hurst_from_a_fractional_brownian_path() {
  let fgn = Fgn::<f64, _>::new(0.3, 4096, Some(1.0), Deterministic::new(7));
  let path = fgn.sample();

  let estimator = RescaledRange {
    take_differences: false,
    ..RescaledRange::default()
  };
  let est = estimator.estimate(path.view()).unwrap();
  assert!((est.hurst - 0.3).abs() < 0.1, "H = {:.3}", est.hurst);
}
