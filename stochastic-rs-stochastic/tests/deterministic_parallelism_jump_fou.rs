//! Reproducibility tests for `JumpFou`'s diffusion fix. `JumpFou` was
//! documented as the crate's one remaining full exception — "no randomness
//! derives from `self.seed` at all" — on the grounds that *both* its
//! private `fgn: Fgn<T, Unseeded, B>` diffusion field and its public
//! `cpoisson: CompoundPoisson<T, D>` jump field were "the type's own
//! public-field-shaped structural pins." That was wrong about `fgn`:
//! unlike `cpoisson`, the `fgn` field is **private**, byte-for-byte the
//! same shape as `JumpFOUCustom`'s field this wave already fixed
//! non-breakingly (see `deterministic_parallelism_jump_fou_custom.rs`).
//! `JumpFou::sampler()` builds its Gaussian source from `self.seed.derive()`
//! directly (rather than `self.fgn.sampler()`, which read `fgn`'s own dead
//! `Unseeded` field) and borrows `fgn` only for its `Arc`-shared FFT
//! plan/eigenvalues.
//!
//! `cpoisson` genuinely was public and structurally pinned to `Unseeded`
//! (that half of the original exception was correct) — fixed by the
//! zero-exception-reproducibility wave's Task 2, the same breaking widening
//! (`CompoundPoisson<T, D>` -> `CompoundPoisson<T, D, S>`, `new()` absorbing
//! the jump-size distribution and a new top-level `lambda: T` field) Task 1
//! applied to `Merton`/`Kou`/`LevyDiffusion` and Task 2 also applied to
//! `Bates1996`. `JumpFou` is now **fully** seed-reproducible — this was the
//! crate's last exception of any kind. See MIGRATION.md.
//!
//! The tests below still separate a zero-intensity diffusion check from a
//! nonzero-intensity check: `JumpFou`'s single `Array1<T>` output mixes the
//! fGn diffusion and the jump term additively at every index (`out[i] = ...
//! + sigma * fgn[i-1] + jump_increments[i]`), so a zero-intensity run proves
//! the diffusion half is seed-driven in isolation (no jump draw at all —
//! `CompoundPoisson::sample_grid_increments` short-circuits to an all-zero
//! array with no RNG draw when `lambda_dt <= 0.0`, confirmed by reading that
//! function), independent of whether the jump half is also fixed. The
//! dedicated full-reproducibility battery (bit-identity at nonzero lambda,
//! thread-count independence at m=64/256, distinctness at m=256) lives in
//! `reproducibility_bates_jump_fou.rs`, alongside `Bates1996`'s.

use ndarray::Array1;
use rayon::ThreadPoolBuilder;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::scalar::ScalarNormal;
use stochastic_rs_stochastic::jump::jump_fou::JumpFou;
use stochastic_rs_stochastic::traits::ProcessExt;

const SEED: u64 = 42;
const N: usize = 128;

fn jump_fou_zero_jump(seed: u64) -> JumpFou<f64, ScalarNormal<f64>, Deterministic> {
  JumpFou::new(
    0.65,
    1.5,
    0.0,
    0.2,
    0.0,
    ScalarNormal::new(0.0, 1.0),
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
fn jump_fou_diffusion_is_seed_reproducible_with_zero_jump_intensity() {
  let a = jump_fou_zero_jump(SEED).sample();
  let b = jump_fou_zero_jump(SEED).sample();
  assert_eq!(
    bits_1d(&a),
    bits_1d(&b),
    "two identically-seeded, zero-jump-intensity JumpFou objects diverged"
  );
}

#[test]
fn jump_fou_diffusion_sample_par_is_thread_count_independent() {
  let m = 64;
  let run =
    |threads: usize| bits_paths(&pool(threads).install(|| jump_fou_zero_jump(SEED).sample_par(m)));

  let r1 = run(1);
  let r3 = run(3);
  let r8 = run(8);

  assert_eq!(r1.len(), m);
  assert_eq!(
    r1, r3,
    "JumpFou sample_par diverged between 1 and 3 threads"
  );
  assert_eq!(
    r1, r8,
    "JumpFou sample_par diverged between 1 and 8 threads"
  );
}

/// Beyond `MAX_CHUNKS = 64`: `JumpFou` goes through `ProcessExt::sample_par`'s
/// default, so several paths share one chunk's derived basis at `m = 256`.
#[test]
fn jump_fou_diffusion_sample_par_is_thread_count_independent_beyond_max_chunks() {
  let m = 256;
  let run =
    |threads: usize| bits_paths(&pool(threads).install(|| jump_fou_zero_jump(SEED).sample_par(m)));

  let r1 = run(1);
  let r8 = run(8);

  assert_eq!(r1.len(), m);
  assert_eq!(
    r1, r8,
    "JumpFou sample_par diverged between 1 and 8 threads at m=256"
  );
}

#[test]
fn jump_fou_diffusion_sample_par_paths_are_distinct() {
  let m = 16;
  let paths = jump_fou_zero_jump(SEED).sample_par(m);
  assert_eq!(paths.len(), m);
  let keys = paths
    .iter()
    .map(bits_1d)
    .collect::<std::collections::HashSet<_>>();
  assert_eq!(keys.len(), m, "JumpFou sample_par produced duplicate paths");
}
