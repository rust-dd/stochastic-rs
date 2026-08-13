// docs: processes#geometric-brownian-motion
//! Backs the GBM example on the processes catalog page.

use stochastic_rs::prelude::*;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::diffusion::gbm::Gbm;

#[test]
fn gbm_sample_and_sample_par() {
  let p = Gbm::<f64, _>::new(
    0.05,
    0.2,
    1_000,
    Some(100.0),
    Some(1.0),
    Deterministic::new(42),
  );
  let path = p.sample();
  assert_eq!(path.len(), 1_000);

  let paths = p.sample_par(10_000);
  assert_eq!(paths.len(), 10_000);
  assert_eq!(paths[0].len(), 1_000);
}
