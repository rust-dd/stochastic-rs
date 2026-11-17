//! The launch seed reaches the kernels whole: a seed's high half moves every path.

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::euler::EulerKernel;

use super::common::Device;

#[test]
fn the_high_half_of_the_launch_seed_reaches_the_kernel() {
  let gbm = Gbm::<f32, _>::new(0.05, 0.2, 64, 100.0, Some(1.0), Deterministic::new(1));
  let paths = |seed: u64| {
    Device::default()
      .euler_kernel(&gbm, 0, 8, seed)
      .expect("a device launch")
  };
  assert_ne!(
    paths(0),
    paths(0x1_0000_0001),
    "seed ^ (seed >> 32) folded both launch seeds onto one 32-bit key"
  );
  assert_eq!(paths(7), paths(7));
}
