//! TDD tests for A1-c Task 1: `Default` on the six flagship distributions.
//! See each type's own `Default` impl doc for where its parameter values
//! come from.

use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::lognormal::SimdLogNormal;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::studentt::SimdStudentT;
use stochastic_rs_distributions::uniform::SimdUniform;

const M: usize = 64;

fn all_finite(mut d: impl FnMut() -> f64) -> bool {
  (0..M).map(|_| d()).all(f64::is_finite)
}

/// Every Default-constructible distribution must sample finite output out
/// of the box.
#[test]
fn defaults_sample_finite() {
  let mut d = SimdNormal::<f64>::default().seeded(&Unseeded);
  assert!(all_finite(|| d.sample()));

  let d = SimdUniform::<f64>::default();
  assert!(all_finite(|| d.sample_fast()));

  let d = SimdExp::<f64>::default();
  assert!(all_finite(|| d.sample_fast()));

  let d = SimdGamma::<f64>::default();
  assert!(all_finite(|| d.sample_fast()));

  let d = SimdLogNormal::<f64>::default();
  assert!(all_finite(|| d.sample_fast()));

  let d = SimdStudentT::<f64>::default();
  assert!(all_finite(|| d.sample_fast()));
}
