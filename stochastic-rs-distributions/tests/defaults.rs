//! TDD tests for A1-c Task 1: `Default` on the six flagship distributions.
//! See each type's own `Default` impl doc for where its parameter values
//! come from.
//!
//! `clone_preserves_deterministic_path`-style seed-replay is intentionally
//! **not** asserted here: `ProcessExt`'s `## Clone semantics` section
//! (`stochastic-rs-stochastic/src/traits/process.rs`) documents that this
//! crate's own `Clone` deliberately diverges from a process's — cloning a
//! distribution re-seeds independently by design (`SimdNormal::clone`'s own
//! doc: "cloning a stochastic source means 'give me an independent
//! stream'"). Asserting bit-for-bit equality here would pin the wrong
//! contract; `clone_reseeds_independently` below pins the actual one.

use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::lognormal::SimdLogNormal;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::studentt::SimdStudentT;
use stochastic_rs_distributions::uniform::SimdUniform;

const M: usize = 64;

fn all_finite(d: impl Fn() -> f64) -> bool {
  (0..M).map(|_| d()).all(f64::is_finite)
}

/// Every Default-constructible distribution must sample finite output out
/// of the box.
#[test]
fn defaults_sample_finite() {
  let d = SimdNormal::<f64>::default();
  assert!(all_finite(|| d.sample_fast()));

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

/// This crate's actual, pre-existing `Clone` contract (unrelated to this
/// task, untouched by it): cloning re-seeds independently rather than
/// snapshotting, so a freshly-cloned `Default` does **not** replay the
/// original's stream — the opposite of `ProcessExt`'s process contract. See
/// the module doc above.
#[test]
fn clone_reseeds_independently() {
  let a = SimdNormal::<f64>::new(0.0, 1.0, &Unseeded);
  let b = a.clone();
  let from_a = (0..M).map(|_| a.sample_fast()).collect::<Vec<f64>>();
  let from_b = (0..M).map(|_| b.sample_fast()).collect::<Vec<f64>>();
  assert_ne!(from_a, from_b);
}
