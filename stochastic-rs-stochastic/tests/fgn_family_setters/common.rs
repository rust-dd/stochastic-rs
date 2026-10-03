use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::normal::SimdNormal;

pub(super) fn seed(value: u64) -> Deterministic {
  Deterministic::new(value)
}

pub(super) fn jump_law(std_dev: f64) -> SimdNormal<f64> {
  SimdNormal::new(0.0, std_dev)
}
