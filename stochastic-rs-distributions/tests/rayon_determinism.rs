//! One `&dist` across rayon workers with per-worker rngs, and per-index forks, give the serial output.

use rayon::prelude::*;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

fn draws(d: &SimdNormal<f64>, seed: u64) -> Vec<u64> {
  use rand::distr::Distribution;
  let mut rng = SimdRng::from_seed(seed);
  (0..1000).map(|_| d.sample(&mut rng).to_bits()).collect()
}

#[test]
fn shared_dist_with_per_worker_rngs_is_deterministic() {
  let d = SimdNormal::<f64>::new(0.0, 1.0);
  let par = (0..8u64)
    .into_par_iter()
    .map(|i| draws(&d, i))
    .collect::<Vec<_>>();
  let serial = (0..8u64).map(|i| draws(&d, i)).collect::<Vec<_>>();
  assert_eq!(par, serial);
}

#[test]
fn per_worker_forks_are_deterministic() {
  let fork = |i: u64| {
    SimdNormal::<f64>::new(0.0, 1.0)
      .seeded(&Deterministic::new(11))
      .fork(i)
      .sample_n(256)
      .iter()
      .map(|x| x.to_bits())
      .collect::<Vec<_>>()
  };
  let par = (0..8u64).into_par_iter().map(fork).collect::<Vec<_>>();
  let serial = (0..8u64).map(fork).collect::<Vec<_>>();
  assert_eq!(par, serial);
  let pool = rayon::ThreadPoolBuilder::new()
    .num_threads(2)
    .build()
    .unwrap();
  assert_eq!(
    pool.install(|| (0..8u64).into_par_iter().map(fork).collect::<Vec<_>>()),
    serial
  );
}
