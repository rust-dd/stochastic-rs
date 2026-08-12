//! Pins the decision made for the deterministic-parallelism wave's Task 3:
//! whole-struct `Clone` on a process **snapshots** its seed strategy rather
//! than forking it. Cloning copies the seed's *current* state into an
//! independent copy, so the original and the clone start their next draw
//! from the identical point — sampled immediately after cloning, with no
//! intervening draw on either side, they replay the same path. See
//! `ProcessExt`'s trait doc for the full rationale and the accepted
//! tradeoff.
//!
//! This is unrelated to the `sampler()`-internal rule documented on
//! `ProcessExt` ("Reproducibility requirement on implementors": a sampler
//! must `derive()`, never `clone()`, its basis from `self.seed`). That rule
//! governs how one process instance builds a *fresh* per-call/per-chunk
//! basis from its own live seed field; this file is about what a caller
//! gets from `process.clone()` itself, once, up front. Every process type
//! in this crate that derives `Clone` today does so via a plain
//! `#[derive(Clone)]` over a `seed: S` field, so it inherits
//! `Deterministic::clone()`'s snapshot behaviour with no type-specific
//! logic to diverge — these tests exercise three structurally different
//! Clone-deriving families (a `Copy` diffusion SDE, a `Copy` noise
//! generator, and a non-`Copy` rough/Markov-lift process) to confirm the
//! guarantee holds uniformly rather than only in the simplest case.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::ou::Ou;
use stochastic_rs_stochastic::noise::gn::Gn;
use stochastic_rs_stochastic::rough::rl_fbm::RlFBm;
use stochastic_rs_stochastic::traits::ProcessExt;

const SEED: u64 = 42;

fn ou(seed: u64) -> Ou<f64, Deterministic> {
  Ou::new(
    0.5,
    0.02,
    0.1,
    32,
    Some(0.03),
    Some(1.0),
    Deterministic::new(seed),
  )
}

fn gn(seed: u64) -> Gn<f64, Deterministic> {
  Gn::new(32, Some(1.0), Deterministic::new(seed))
}

fn rl_fbm(seed: u64) -> RlFBm<f64, Deterministic> {
  RlFBm::new(0.25, 64, Some(1.0), None, Deterministic::new(seed))
}

fn bits(path: &Array1<f64>) -> Vec<u64> {
  path.iter().map(|x| x.to_bits()).collect()
}

/// Cloning right after construction, then sampling the original and the
/// clone once each, replays the identical path.
#[test]
fn ou_clone_replays_identical_path() {
  let a = ou(SEED);
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));
}

/// Same guarantee on a `Copy` noise-generator type, structurally distinct
/// from the diffusion-SDE family above.
#[test]
fn gn_clone_replays_identical_path() {
  let a = gn(SEED);
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));
}

/// Same guarantee on a non-`Copy` process carrying an internal precomputed
/// cache (`RlFBm`'s `MarkovLift`) alongside the seed field.
#[test]
fn rl_fbm_clone_replays_identical_path() {
  let a = rl_fbm(SEED);
  let b = a.clone();
  assert_eq!(bits(&a.sample()), bits(&b.sample()));
}

/// The snapshot is of the seed's *current* state, not its original
/// construction-time value: sampling the original once before cloning
/// moves the point the clone goes on to replay from.
#[test]
fn clone_snapshots_current_state_not_construction_seed() {
  let a = ou(SEED);
  let first = a.sample();
  let b = a.clone();
  let second_from_a = a.sample();
  let first_from_b = b.sample();
  assert_ne!(
    bits(&first),
    bits(&second_from_a),
    "sampling twice from the same object must advance, not replay"
  );
  assert_eq!(
    bits(&second_from_a),
    bits(&first_from_b),
    "a clone taken after one sample must replay from THAT state, not the original construction seed"
  );
}

/// After the snapshot is taken, the two objects are fully independent:
/// sampling the original does not perturb what the clone goes on to
/// produce.
#[test]
fn clone_is_independent_of_original_after_snapshot() {
  let a = ou(SEED);
  let b = a.clone();
  let baseline = ou(SEED).sample();
  let _ = a.sample();
  let from_b = b.sample();
  assert_eq!(
    bits(&baseline),
    bits(&from_b),
    "advancing the original after cloning must not affect the clone's own independent stream"
  );
}
