use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::scalar::ScalarNormal;

pub(super) fn seed(value: u64) -> Deterministic {
  Deterministic::new(value)
}

pub(super) fn jump_law(std_dev: f64) -> ScalarNormal<f64> {
  ScalarNormal::new(0.0, std_dev)
}
