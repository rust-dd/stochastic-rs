//! `Fgn`/`Fbm` override `ProcessExt::sample_par` to reach the batched
//! backend path (`Backend::generate_batch`) instead of the trait default's
//! `chunked_samplers` — Task 1's fix (`deterministic_parallelism.rs`) never
//! touched them, and a reviewer found an independent instance of the same
//! nondeterminism class there: `Cpu::generate_batch` used to do
//! `(0..m).into_par_iter().map(|_| fgn.sample_cpu())`, and `sample_cpu`
//! reads `&self.seed` — a shared `Deterministic` atomic — fresh, from
//! *inside* the parallel region, once per path. Which of the `m` parallel
//! iterations claimed which tick of that atomic depended on rayon's
//! scheduling, which depends on thread-pool size.
//!
//! `Fbm::sample_par` carried a second, deeper bug the thread-count-only
//! framing above does not capture: it drove the batch through
//! `self.fgn.noise_batch(m)`, where `self.fgn: Fgn<T, Unseeded, B>` is
//! *always* `Unseeded` by construction (see `Fbm`'s own doc) — so the
//! batch never read `Fbm`'s own `self.seed` at all, seeded or not. A
//! `Deterministic`-seeded `Fbm::sample_par` used to draw fresh randomness on
//! every single call. `fbm_sample_par_is_seed_reproducible` below is the
//! test that catches this specifically (thread-count independence alone
//! would not: a function that ignores its seed is trivially "independent"
//! of thread count too).

use ndarray::Array1;
use rayon::ThreadPoolBuilder;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::noise::fgn::Fgn;
use stochastic_rs_stochastic::process::fbm::Fbm;
use stochastic_rs_stochastic::traits::ProcessExt;

const SEED: u64 = 42;

fn fgn(seed: u64) -> Fgn<f64, Deterministic> {
  Fgn::<f64, _>::new(0.7, 128, Some(1.0), Deterministic::new(seed))
}

fn fbm(seed: u64) -> Fbm<f64, Deterministic> {
  Fbm::<f64, _>::new(0.7, 128, Some(1.0), Deterministic::new(seed))
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

/// `Fgn::sample_par` must be bit-identical across thread-pool sizes.
#[test]
fn fgn_sample_par_is_thread_count_independent() {
  let m = 64;
  let run = |threads: usize| bits_paths(&pool(threads).install(|| fgn(SEED).sample_par(m)));

  let r1 = run(1);
  let r3 = run(3);
  let r8 = run(8);

  assert_eq!(r1.len(), m);
  assert_eq!(r1, r3, "Fgn sample_par diverged between 1 and 3 threads");
  assert_eq!(r1, r8, "Fgn sample_par diverged between 1 and 8 threads");
}

/// Same guarantee, beyond `MAX_CHUNKS = 64`: several paths then share one
/// chunk's `SimdNormal`, the regime `m = 64` above cannot reach.
#[test]
fn fgn_sample_par_is_thread_count_independent_beyond_max_chunks() {
  let m = 256;
  let run = |threads: usize| bits_paths(&pool(threads).install(|| fgn(SEED).sample_par(m)));

  let r1 = run(1);
  let r8 = run(8);

  assert_eq!(r1.len(), m);
  assert_eq!(
    r1, r8,
    "Fgn sample_par diverged between 1 and 8 threads at m=256"
  );
}

/// `Fbm::sample_par` must be bit-identical across thread-pool sizes.
#[test]
fn fbm_sample_par_is_thread_count_independent() {
  let m = 64;
  let run = |threads: usize| bits_paths(&pool(threads).install(|| fbm(SEED).sample_par(m)));

  let r1 = run(1);
  let r3 = run(3);
  let r8 = run(8);

  assert_eq!(r1.len(), m);
  assert_eq!(r1, r3, "Fbm sample_par diverged between 1 and 3 threads");
  assert_eq!(r1, r8, "Fbm sample_par diverged between 1 and 8 threads");
}

#[test]
fn fbm_sample_par_is_thread_count_independent_beyond_max_chunks() {
  let m = 256;
  let run = |threads: usize| bits_paths(&pool(threads).install(|| fbm(SEED).sample_par(m)));

  let r1 = run(1);
  let r8 = run(8);

  assert_eq!(r1.len(), m);
  assert_eq!(
    r1, r8,
    "Fbm sample_par diverged between 1 and 8 threads at m=256"
  );
}

/// The bug specific to `Fbm`: its embedded `fgn` is hard-wired `Unseeded`,
/// so `sample_par` used to draw fresh randomness on every call regardless of
/// `self.seed`. Two identically-`Deterministic`-seeded `Fbm`s must now
/// agree, proving the batch actually consults the real seed — thread-count
/// independence alone cannot show this (an `Unseeded`-driven batch is
/// trivially "independent" of thread count too, just never reproducible).
#[test]
fn fbm_sample_par_is_seed_reproducible() {
  let m = 32;
  let a = bits_paths(&fbm(SEED).sample_par(m));
  let b = bits_paths(&fbm(SEED).sample_par(m));
  assert_eq!(
    a, b,
    "two identically-seeded Fbm objects diverged on sample_par"
  );
}

/// Same proof for `Fgn`, whose batch already read the correct seed source
/// (the race, not the seed source, was `Fgn`'s bug) — kept for symmetry and
/// to guard against a future regression conflating the two.
#[test]
fn fgn_sample_par_is_seed_reproducible() {
  let m = 32;
  let a = bits_paths(&fgn(SEED).sample_par(m));
  let b = bits_paths(&fgn(SEED).sample_par(m));
  assert_eq!(
    a, b,
    "two identically-seeded Fgn objects diverged on sample_par"
  );
}

/// `m = 16`: no two `Fgn` paths identical. A chunking bug that let two
/// chunks reuse the same basis would collapse chunk 0's paths onto chunk
/// 1's.
#[test]
fn fgn_sample_par_paths_are_distinct() {
  let m = 16;
  let paths = fgn(SEED).sample_par(m);
  assert_eq!(paths.len(), m);
  let keys = paths
    .iter()
    .map(bits_1d)
    .collect::<std::collections::HashSet<_>>();
  assert_eq!(keys.len(), m, "Fgn sample_par produced duplicate paths");
}

#[test]
fn fbm_sample_par_paths_are_distinct() {
  let m = 16;
  let paths = fbm(SEED).sample_par(m);
  assert_eq!(paths.len(), m);
  let keys = paths
    .iter()
    .map(bits_1d)
    .collect::<std::collections::HashSet<_>>();
  assert_eq!(keys.len(), m, "Fbm sample_par produced duplicate paths");
}

/// `m = 0` returns an empty `Vec` and `m = 1` returns exactly one path;
/// neither panics, for either type.
#[test]
fn fgn_fbm_sample_par_degenerate_m() {
  assert_eq!(fgn(SEED).sample_par(0).len(), 0);
  assert_eq!(fgn(SEED).sample_par(1).len(), 1);
  assert_eq!(fbm(SEED).sample_par(0).len(), 0);
  assert_eq!(fbm(SEED).sample_par(1).len(), 1);
}
