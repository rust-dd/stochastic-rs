// docs: concepts/seeding#simdrngext-generic-backing-rng
//! Backs the single- vs dual-stream `SimdNormal` example on the seeding
//! concept page. `SimdNormalDual` only exists under `dual-stream-rng`,
//! so the whole file is gated on that feature.

#![cfg(feature = "dual-stream-rng")]

use stochastic_rs::distributions::SimdNormalDual;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::traits::DistributionExt;

#[test]
fn single_and_dual_stream_normal_agree_on_moments() {
  let n = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(42));
  let n_dual = SimdNormalDual::<f64>::new(0.0, 1.0, &Deterministic::new(42));

  assert!((n.mean() - n_dual.mean()).abs() < 1e-12);
  assert!((n.variance() - n_dual.variance()).abs() < 1e-12);
}
