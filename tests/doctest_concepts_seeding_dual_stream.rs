// docs: concepts/seeding#simdrngext-generic-backing-rng
//! Backs the single- vs dual-stream `SimdNormal` example on the seeding
//! concept page. `SimdRngDual` only exists under `unstable-dual-stream-rng`,
//! so the whole file is gated on that feature.

#![cfg(feature = "unstable-dual-stream-rng")]

use stochastic_rs::distributions::Seeded;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::traits::DistributionExt;
use stochastic_rs_core::simd_rng_dual::SimdRngDual;

#[test]
fn single_and_dual_stream_normal_agree_on_moments() {
  let n = SimdNormal::<f64>::new(0.0, 1.0);
  let n_dual = Seeded::<_, SimdRngDual>::new(n, &Deterministic::new(42));

  assert!((n.mean() - n_dual.dist().mean()).abs() < 1e-12);
  assert!((n.variance().unwrap() - n_dual.dist().variance().unwrap()).abs() < 1e-12);
}
