//! Reproducibility tests for Task 1 of the zero-exception-reproducibility
//! wave: `Merton`, `Kou`, `LevyDiffusion` absorb their compound-Poisson jump
//! driver's construction into `new()`, threading the constructor's own
//! `seed: S` into it (`seed.clone().derive()`) instead of hard-wiring
//! `cpoisson: CompoundPoisson<T, D>` to `Unseeded` in the public field type.
//! All three are now fully seed-reproducible — no exception to
//! `ProcessExt`'s reproducibility guarantee. See `traits/process.rs`'s
//! trait-level reproducibility section.
//!
//! Before this fix (see this file's prior revision / the RED-phase probe in
//! the task report), two identically-`Deterministic`-seeded objects
//! diverged under any nonzero jump intensity, because `cpoisson`'s own seed
//! field was structurally pinned to `Unseeded` regardless of the outer
//! `S`. `lambda = 50` below makes jumps dominate the path, so a
//! reproducibility failure in the jump component cannot hide behind a
//! diffusion-only comparison.

use ndarray::Array1;
use rayon::ThreadPoolBuilder;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::scalar::ScalarNormal;
use stochastic_rs_stochastic::jump::kou::Kou;
use stochastic_rs_stochastic::jump::levy_diffusion::LevyDiffusion;
use stochastic_rs_stochastic::jump::merton::Merton;
use stochastic_rs_stochastic::traits::ProcessExt;

const SEED: u64 = 42;
const N: usize = 128;
const LAMBDA: f64 = 50.0;

fn merton_with_jumps(seed: u64) -> Merton<f64, ScalarNormal<f64>, Deterministic> {
  Merton::new(
    0.03,
    0.2,
    LAMBDA,
    0.0,
    ScalarNormal::new(0.0, 0.1),
    N,
    Some(0.0),
    Some(1.0),
    Deterministic::new(seed),
  )
}

fn kou_with_jumps(seed: u64) -> Kou<f64, ScalarNormal<f64>, Deterministic> {
  Kou::new(
    0.03,
    0.2,
    LAMBDA,
    0.0,
    ScalarNormal::new(0.0, 0.12),
    N,
    Some(0.0),
    Some(1.0),
    Deterministic::new(seed),
  )
}

fn levy_diffusion_with_jumps(seed: u64) -> LevyDiffusion<f64, ScalarNormal<f64>, Deterministic> {
  LevyDiffusion::new(
    0.01,
    0.2,
    LAMBDA,
    ScalarNormal::new(0.0, 0.08),
    N,
    Some(0.0),
    Some(1.0),
    Deterministic::new(seed),
  )
}

fn bits_1d(path: &Array1<f64>) -> Vec<u64> {
  path.iter().map(|x| x.to_bits()).collect()
}

fn bits_paths(paths: &[Array1<f64>]) -> Vec<Vec<u64>> {
  paths.iter().map(bits_1d).collect()
}

fn pool(num_threads: usize) -> rayon::ThreadPool {
  ThreadPoolBuilder::new()
    .num_threads(num_threads)
    .build()
    .expect("failed to build rayon thread pool")
}

#[test]
fn merton_is_seed_reproducible_including_jumps() {
  let a = merton_with_jumps(SEED).sample();
  let b = merton_with_jumps(SEED).sample();
  assert_eq!(
    bits_1d(&a),
    bits_1d(&b),
    "two identically-seeded Merton objects diverged under lambda={LAMBDA}"
  );
}

#[test]
fn merton_sample_par_is_thread_count_independent() {
  for &m in &[64usize, 256usize] {
    let run =
      |threads: usize| bits_paths(&pool(threads).install(|| merton_with_jumps(SEED).sample_par(m)));
    let r1 = run(1);
    let r3 = run(3);
    let r8 = run(8);
    assert_eq!(r1.len(), m);
    assert_eq!(
      r1, r3,
      "Merton sample_par diverged between 1 and 3 threads at m={m}"
    );
    assert_eq!(
      r1, r8,
      "Merton sample_par diverged between 1 and 8 threads at m={m}"
    );
  }
}

#[test]
fn merton_sample_par_paths_are_distinct() {
  let m = 256;
  let paths = merton_with_jumps(SEED).sample_par(m);
  assert_eq!(paths.len(), m);
  let keys = paths
    .iter()
    .map(bits_1d)
    .collect::<std::collections::HashSet<_>>();
  assert_eq!(
    keys.len(),
    m,
    "Merton sample_par produced duplicate paths at m=256"
  );
}

#[test]
fn kou_is_seed_reproducible_including_jumps() {
  let a = kou_with_jumps(SEED).sample();
  let b = kou_with_jumps(SEED).sample();
  assert_eq!(
    bits_1d(&a),
    bits_1d(&b),
    "two identically-seeded Kou objects diverged under lambda={LAMBDA}"
  );
}

#[test]
fn kou_sample_par_is_thread_count_independent() {
  for &m in &[64usize, 256usize] {
    let run =
      |threads: usize| bits_paths(&pool(threads).install(|| kou_with_jumps(SEED).sample_par(m)));
    let r1 = run(1);
    let r3 = run(3);
    let r8 = run(8);
    assert_eq!(r1.len(), m);
    assert_eq!(
      r1, r3,
      "Kou sample_par diverged between 1 and 3 threads at m={m}"
    );
    assert_eq!(
      r1, r8,
      "Kou sample_par diverged between 1 and 8 threads at m={m}"
    );
  }
}

#[test]
fn kou_sample_par_paths_are_distinct() {
  let m = 256;
  let paths = kou_with_jumps(SEED).sample_par(m);
  assert_eq!(paths.len(), m);
  let keys = paths
    .iter()
    .map(bits_1d)
    .collect::<std::collections::HashSet<_>>();
  assert_eq!(
    keys.len(),
    m,
    "Kou sample_par produced duplicate paths at m=256"
  );
}

#[test]
fn levy_diffusion_is_seed_reproducible_including_jumps() {
  let a = levy_diffusion_with_jumps(SEED).sample();
  let b = levy_diffusion_with_jumps(SEED).sample();
  assert_eq!(
    bits_1d(&a),
    bits_1d(&b),
    "two identically-seeded LevyDiffusion objects diverged under lambda={LAMBDA}"
  );
}

#[test]
fn levy_diffusion_sample_par_is_thread_count_independent() {
  for &m in &[64usize, 256usize] {
    let run = |threads: usize| {
      bits_paths(&pool(threads).install(|| levy_diffusion_with_jumps(SEED).sample_par(m)))
    };
    let r1 = run(1);
    let r3 = run(3);
    let r8 = run(8);
    assert_eq!(r1.len(), m);
    assert_eq!(
      r1, r3,
      "LevyDiffusion sample_par diverged between 1 and 3 threads at m={m}"
    );
    assert_eq!(
      r1, r8,
      "LevyDiffusion sample_par diverged between 1 and 8 threads at m={m}"
    );
  }
}

#[test]
fn levy_diffusion_sample_par_paths_are_distinct() {
  let m = 256;
  let paths = levy_diffusion_with_jumps(SEED).sample_par(m);
  assert_eq!(paths.len(), m);
  let keys = paths
    .iter()
    .map(bits_1d)
    .collect::<std::collections::HashSet<_>>();
  assert_eq!(
    keys.len(),
    m,
    "LevyDiffusion sample_par produced duplicate paths at m=256"
  );
}

/// Guards the single-source-of-truth invariant a Task 1 review caught
/// broken on `Merton` (`with_cpoisson` silently dropping the driver's
/// lambda — see `with_setters_merton.rs`'s dedicated regression tests):
/// `self.lambda` and the otherwise-cosmetic mirror `cpoisson.poisson.lambda`
/// must agree right out of `new()`, for all three types, including
/// `LevyDiffusion`, which gained its own top-level `lambda` field as part
/// of that same fix (previously `self.lambda` did not exist on
/// `LevyDiffusion` at all).
#[test]
fn jump_family_lambda_is_single_sourced_at_construction() {
  let m = merton_with_jumps(SEED);
  assert_eq!(m.lambda, LAMBDA);
  assert_eq!(m.cpoisson.poisson.lambda, LAMBDA);

  let k = kou_with_jumps(SEED);
  assert_eq!(k.lambda, LAMBDA);
  assert_eq!(k.cpoisson.poisson.lambda, LAMBDA);

  let l = levy_diffusion_with_jumps(SEED);
  assert_eq!(l.lambda, LAMBDA);
  assert_eq!(l.cpoisson.poisson.lambda, LAMBDA);
}
