//! Reproducibility tests for Task 2 of the zero-exception-reproducibility
//! wave: `Bates1996` and `JumpFou` absorb their compound-Poisson jump
//! driver's construction into `new()`, threading the constructor's own
//! `seed: S` into it (`seed.clone().derive()`) instead of hard-wiring
//! `cpoisson: CompoundPoisson<T, D>` to `Unseeded` in the public field type
//! — the same fix Task 1 applied to `Merton`/`Kou`/`LevyDiffusion` (see
//! `reproducibility_jump_family.rs`). Both types' diffusion component was
//! already seed-reproducible before this task (`Bates1996`'s `cgns`, fixed
//! by an earlier wave; `JumpFou`'s private `fgn`, fixed by the predecessor
//! wave's final round — see `deterministic_parallelism_jump_fou.rs`), so
//! this task closes the crate's last exception: **zero** exceptions remain.
//!
//! Before this fix, two identically-`Deterministic`-seeded objects diverged
//! on their jump-carrying output under any nonzero jump intensity, because
//! `cpoisson`'s own seed field was structurally pinned to `Unseeded`
//! regardless of the outer `S`. `lambda = 50` below makes jumps dominate
//! (mean ~50 arrivals over the unit horizon at `N = 128`), so a
//! reproducibility failure in the jump component cannot hide behind a
//! diffusion-only comparison — confirmed directly by
//! `bates_price_path_diverges_from_zero_lambda_counterfactual` /
//! `jump_fou_diverges_from_zero_lambda_counterfactual` below, which are not
//! merely restating the bit-identity tests: they show the *value* actually
//! moves when the jump component is toggled off, so the bit-identity tests
//! above them cannot be silently passing because the jump term is a
//! structural no-op.

use ndarray::Array1;
use rayon::ThreadPoolBuilder;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::scalar::ScalarNormal;
use stochastic_rs_stochastic::jump::bates::Bates1996;
use stochastic_rs_stochastic::jump::jump_fou::JumpFou;
use stochastic_rs_stochastic::traits::ProcessExt;

const SEED: u64 = 42;
const N: usize = 128;
const LAMBDA: f64 = 50.0;

/// `k = 0` isolates the jump half: the drift's `-lambda*k` compensator term
/// is neutralized regardless of `lambda`'s value, so any observed
/// difference between two `lambda` settings (or two identically-seeded
/// objects) can only come from the jump increments themselves, not a
/// residual deterministic drift shift — the same isolation technique the
/// task's own RED-phase measurement used (see the task report).
fn bates_with_jumps(seed: u64) -> Bates1996<f64, ScalarNormal<f64>, Deterministic> {
  Bates1996::new(
    Some(0.05),
    None,
    None,
    None,
    LAMBDA,
    0.0,
    0.04,
    1.5,
    0.3,
    -0.6,
    ScalarNormal::new(0.0, 0.05),
    N,
    Some(100.0),
    Some(0.04),
    Some(1.0),
    Some(false),
    Deterministic::new(seed),
  )
}

fn bates_lambda(lambda: f64, seed: u64) -> Bates1996<f64, ScalarNormal<f64>, Deterministic> {
  Bates1996::new(
    Some(0.05),
    None,
    None,
    None,
    lambda,
    0.0,
    0.04,
    1.5,
    0.3,
    -0.6,
    ScalarNormal::new(0.0, 0.05),
    N,
    Some(100.0),
    Some(0.04),
    Some(1.0),
    Some(false),
    Deterministic::new(seed),
  )
}

fn jump_fou_with_jumps(seed: u64) -> JumpFou<f64, ScalarNormal<f64>, Deterministic> {
  JumpFou::new(
    0.65,
    1.5,
    0.0,
    0.2,
    LAMBDA,
    ScalarNormal::new(0.0, 0.05),
    N,
    Some(0.0),
    Some(1.0),
    Deterministic::new(seed),
  )
}

fn jump_fou_lambda(lambda: f64, seed: u64) -> JumpFou<f64, ScalarNormal<f64>, Deterministic> {
  JumpFou::new(
    0.65,
    1.5,
    0.0,
    0.2,
    lambda,
    ScalarNormal::new(0.0, 0.05),
    N,
    Some(0.0),
    Some(1.0),
    Deterministic::new(seed),
  )
}

fn bits_1d(path: &Array1<f64>) -> Vec<u64> {
  path.iter().map(|x| x.to_bits()).collect()
}

fn bits_2d([s, v]: &[Array1<f64>; 2]) -> Vec<u64> {
  s.iter().chain(v.iter()).map(|x| x.to_bits()).collect()
}

fn bits_paths_1d(paths: &[Array1<f64>]) -> Vec<Vec<u64>> {
  paths.iter().map(bits_1d).collect()
}

fn bits_paths_2d(paths: &[[Array1<f64>; 2]]) -> Vec<Vec<u64>> {
  paths.iter().map(bits_2d).collect()
}

fn pool(num_threads: usize) -> rayon::ThreadPool {
  ThreadPoolBuilder::new()
    .num_threads(num_threads)
    .build()
    .expect("failed to build rayon thread pool")
}

#[test]
fn bates_is_seed_reproducible_including_jumps() {
  let a = bates_with_jumps(SEED).sample();
  let b = bates_with_jumps(SEED).sample();
  assert_eq!(
    bits_2d(&a),
    bits_2d(&b),
    "two identically-seeded Bates1996 objects diverged under lambda={LAMBDA}"
  );
}

/// Proves the golden-adjacent claim "this is not a diffusion-only
/// comparison": the price path actually moves when jumps are switched off,
/// so `bates_is_seed_reproducible_including_jumps` above is not vacuously
/// passing because `s` never depended on the jump term in the first place.
#[test]
fn bates_price_path_diverges_from_zero_lambda_counterfactual() {
  let [s_jumps, _] = bates_with_jumps(SEED).sample();
  let [s_zero, _] = bates_lambda(0.0, SEED).sample();
  assert_ne!(
    bits_1d(&s_jumps),
    bits_1d(&s_zero),
    "Bates1996 price path at lambda={LAMBDA} must differ from the lambda=0 counterfactual"
  );
}

#[test]
fn bates_sample_par_is_thread_count_independent() {
  for &m in &[64usize, 256usize] {
    let run = |threads: usize| {
      bits_paths_2d(&pool(threads).install(|| bates_with_jumps(SEED).sample_par(m)))
    };
    let r1 = run(1);
    let r3 = run(3);
    let r8 = run(8);
    assert_eq!(r1.len(), m);
    assert_eq!(
      r1, r3,
      "Bates1996 sample_par diverged between 1 and 3 threads at m={m}"
    );
    assert_eq!(
      r1, r8,
      "Bates1996 sample_par diverged between 1 and 8 threads at m={m}"
    );
  }
}

#[test]
fn bates_sample_par_paths_are_distinct() {
  let m = 256;
  let paths = bates_with_jumps(SEED).sample_par(m);
  assert_eq!(paths.len(), m);
  let keys = paths
    .iter()
    .map(bits_2d)
    .collect::<std::collections::HashSet<_>>();
  assert_eq!(
    keys.len(),
    m,
    "Bates1996 sample_par produced duplicate paths at m=256"
  );
}

#[test]
fn jump_fou_is_seed_reproducible_including_jumps() {
  let a = jump_fou_with_jumps(SEED).sample();
  let b = jump_fou_with_jumps(SEED).sample();
  assert_eq!(
    bits_1d(&a),
    bits_1d(&b),
    "two identically-seeded JumpFou objects diverged under lambda={LAMBDA}"
  );
}

/// Same "not a diffusion-only pin" proof as `Bates1996`'s counterpart above,
/// adapted to `JumpFou`'s single, additively-mixed output array.
#[test]
fn jump_fou_diverges_from_zero_lambda_counterfactual() {
  let with_jumps = jump_fou_with_jumps(SEED).sample();
  let zero = jump_fou_lambda(0.0, SEED).sample();
  assert_ne!(
    bits_1d(&with_jumps),
    bits_1d(&zero),
    "JumpFou output at lambda={LAMBDA} must differ from the lambda=0 counterfactual"
  );
}

#[test]
fn jump_fou_sample_par_is_thread_count_independent() {
  for &m in &[64usize, 256usize] {
    let run = |threads: usize| {
      bits_paths_1d(&pool(threads).install(|| jump_fou_with_jumps(SEED).sample_par(m)))
    };
    let r1 = run(1);
    let r3 = run(3);
    let r8 = run(8);
    assert_eq!(r1.len(), m);
    assert_eq!(
      r1, r3,
      "JumpFou sample_par diverged between 1 and 3 threads at m={m}"
    );
    assert_eq!(
      r1, r8,
      "JumpFou sample_par diverged between 1 and 8 threads at m={m}"
    );
  }
}

#[test]
fn jump_fou_sample_par_paths_are_distinct() {
  let m = 256;
  let paths = jump_fou_with_jumps(SEED).sample_par(m);
  assert_eq!(paths.len(), m);
  let keys = paths
    .iter()
    .map(bits_1d)
    .collect::<std::collections::HashSet<_>>();
  assert_eq!(
    keys.len(),
    m,
    "JumpFou sample_par produced duplicate paths at m=256"
  );
}

/// Guards the single-source-of-truth invariant established at construction
/// for both types (see `Bates1996::lambda`'s and `JumpFou::lambda`'s field
/// docs): `self.lambda` and the otherwise-cosmetic mirror
/// `cpoisson.poisson.lambda` must agree right out of `new()`.
#[test]
fn lambda_is_single_sourced_at_construction() {
  let b = bates_with_jumps(SEED);
  assert_eq!(b.lambda, LAMBDA);
  assert_eq!(b.cpoisson.poisson.lambda, LAMBDA);

  let j = jump_fou_with_jumps(SEED);
  assert_eq!(j.lambda, LAMBDA);
  assert_eq!(j.cpoisson.poisson.lambda, LAMBDA);
}
