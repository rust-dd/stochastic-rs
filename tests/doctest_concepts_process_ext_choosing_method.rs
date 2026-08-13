// docs: concepts/process-ext#choosing-a-method
//! Backs the "choosing a method" example on the ProcessExt concept page.

use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::diffusion::gbm::Gbm;
use stochastic_rs::traits::ProcessExt;

#[test]
fn sample_sample_map_and_sample_par() {
  let gbm = Gbm::<f64, _>::new(0.05, 0.2, 64, Some(1.0), Some(1.0), Unseeded);

  // one path
  let path = gbm.sample();
  assert_eq!(path.len(), 64);

  // parallel reduction — no per-path allocation kept
  let mean = gbm
    .sample_map(10_000, |p| (p.last().unwrap() - 1.0).max(0.0))
    .iter()
    .sum::<f64>()
    / 10_000.0;
  assert!(mean >= 0.0);

  // kept paths, for plotting / path-dependent post-processing
  let paths = gbm.sample_par(1_000);
  assert_eq!(paths.len(), 1_000);
}
