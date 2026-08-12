//! `ProcessExt::sample_par` / `sample_map` must be reproducible under
//! `Deterministic` seeding regardless of the ambient rayon thread-pool size.
//!
//! The defect this covers: the pre-fix default methods built one sampler per
//! `map_init` invocation, and rayon decides how many times `map_init`'s init
//! closure fires based on work-stealing, not on `m`. Each sampler
//! construction advances the process's shared `Deterministic` seed state, so
//! the number and interleaving of those advances — and hence the output —
//! depended on scheduling, which in turn depends on the thread-pool size. A
//! reviewer measured three runs of one pinned config giving means 4.01900 /
//! 4.02577 / 4.00938. These tests pin an explicit pool size (1, 3, 8, ...)
//! via `ThreadPoolBuilder::install` for exactly this reason: asserting
//! reproducibility only under the ambient pool would not exercise the
//! defect at all.
//!
//! A follow-up review found the first version of this fix actively
//! regressed processes whose `sampler()` *clones* the seed (`Sabr` and 28
//! others — a non-advancing snapshot per `SeedExt`'s design, so every chunk
//! cloned the same state and `m` extra paths bought zero extra diversity).
//! `sabr_sample_par_is_thread_count_independent` and
//! `sabr_sample_par_paths_are_distinct` below cover that class explicitly,
//! in addition to `Gbm`. The same review also found the fix was a no-op for
//! processes whose `sampler()` reads the seed *lazily* per path (`Heston`
//! and 11 others); that class needs a real sampler rewrite, covered
//! separately.

use ndarray::Array1;
use rayon::ThreadPoolBuilder;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::sabr::Sabr;

const SEED: u64 = 42;

fn gbm(seed: u64) -> Gbm<f64, Deterministic> {
  Gbm::<f64, _>::new(
    0.05,
    0.2,
    32,
    Some(1.0),
    Some(1.0),
    Deterministic::new(seed),
  )
}

/// `Sabr` is "clone-snapshot": `sampler()` does `seed: self.seed.clone()`,
/// and `Deterministic::clone()` is a non-advancing snapshot of the current
/// state — every chunk that clones before `self.seed` itself advances gets
/// the identical snapshot.
fn sabr(seed: u64) -> Sabr<f64, Deterministic> {
  Sabr::<f64, _>::new(
    0.3,
    0.5,
    0.0,
    32,
    Some(1.0),
    Some(0.2),
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

fn bits_2d([a, b]: &[Array1<f64>; 2]) -> Vec<u64> {
  a.iter().chain(b.iter()).map(|x| x.to_bits()).collect()
}

fn pool(num_threads: usize) -> rayon::ThreadPool {
  ThreadPoolBuilder::new()
    .num_threads(num_threads)
    .build()
    .expect("failed to build rayon thread pool")
}

/// `sample_par` must be bit-identical across thread-pool sizes — the defect
/// this task fixes produced means 4.01900 / 4.02577 / 4.00938 for one pinned
/// config.
#[test]
fn sample_par_is_thread_count_independent() {
  let m = 64;
  let run = |threads: usize| bits_paths(&pool(threads).install(|| gbm(SEED).sample_par(m)));

  let r1 = run(1);
  let r3 = run(3);
  let r8 = run(8);

  assert_eq!(r1.len(), m);
  assert_eq!(r1, r3, "sample_par diverged between 1 and 3 threads");
  assert_eq!(r1, r8, "sample_par diverged between 1 and 8 threads");
}

/// Same guarantee, on the "clone-snapshot" class (`Sabr`): pre-fix, every
/// chunk cloned the identical snapshot, so this held trivially (all chunks
/// agreed — with each other, uselessly) but `sabr_sample_par_paths_are_distinct`
/// below would have failed.
#[test]
fn sabr_sample_par_is_thread_count_independent() {
  let m = 64;
  let run = |threads: usize| {
    pool(threads)
      .install(|| sabr(SEED).sample_par(m))
      .iter()
      .map(bits_2d)
      .collect::<Vec<_>>()
  };

  let r1 = run(1);
  let r3 = run(3);
  let r8 = run(8);

  assert_eq!(r1.len(), m);
  assert_eq!(r1, r3, "Sabr sample_par diverged between 1 and 3 threads");
  assert_eq!(r1, r8, "Sabr sample_par diverged between 1 and 8 threads");
}

/// `Sabr` (clone-snapshot), `m = 64`: no two paths identical. This is the
/// test that catches the clone-snapshot regression specifically — with
/// `chunk_count` capping chunks at `MAX_CHUNKS = 64` and every chunk cloning
/// the same unchanged seed, `sample_par(m)` would degenerate to at most
/// `MAX_CHUNKS` distinct paths repeated, independent of `m`, without
/// `advance_chunk_seed` giving each chunk's clone a different state to
/// snapshot.
#[test]
fn sabr_sample_par_paths_are_distinct() {
  let m = 64;
  let paths = sabr(SEED).sample_par(m);
  assert_eq!(paths.len(), m);
  let keys = paths
    .iter()
    .map(bits_2d)
    .collect::<std::collections::HashSet<_>>();
  assert_eq!(keys.len(), m, "Sabr sample_par produced duplicate paths");
}

/// Two identically-seeded processes, same `m`, run under *different*
/// explicit pool sizes ⇒ identical output. This also covers the plain
/// same-object / same-pool repeatability the task brief asks for, one level
/// stronger (pool sizes differ between the two calls, not just the objects).
#[test]
fn sample_par_is_repeatable_across_objects() {
  let m = 48;
  let a = bits_paths(&pool(2).install(|| gbm(SEED).sample_par(m)));
  let b = bits_paths(&pool(5).install(|| gbm(SEED).sample_par(m)));
  assert_eq!(a, b, "two identically-seeded objects diverged");
}

/// Same guarantee as `sample_par_is_thread_count_independent`, through
/// `sample_map` with a summary closure instead of the raw paths.
#[test]
fn sample_map_is_thread_count_independent() {
  let m = 64;
  let summarize = |p: &Array1<f64>| *p.last().expect("path must be non-empty");
  let run = |threads: usize| {
    pool(threads)
      .install(|| gbm(SEED).sample_map(m, summarize))
      .iter()
      .map(|x| x.to_bits())
      .collect::<Vec<_>>()
  };

  let r1 = run(1);
  let r4 = run(4);
  let r8 = run(8);

  assert_eq!(r1.len(), m);
  assert_eq!(r1, r4, "sample_map diverged between 1 and 4 threads");
  assert_eq!(r1, r8, "sample_map diverged between 1 and 8 threads");
}

/// `m = 16`: no two paths are identical. A chunking bug that let two chunks
/// consume the same seed (instead of advancing it once per chunk) would
/// collapse chunk 0's paths onto chunk 1's.
#[test]
fn sample_par_paths_are_distinct() {
  let m = 16;
  let paths = gbm(SEED).sample_par(m);
  assert_eq!(paths.len(), m);
  for i in 0..paths.len() {
    for j in (i + 1)..paths.len() {
      assert_ne!(
        bits_1d(&paths[i]),
        bits_1d(&paths[j]),
        "paths {i} and {j} are identical"
      );
    }
  }
}

/// `m = 0` returns an empty `Vec` and `m = 1` returns exactly one path;
/// neither panics.
#[test]
fn sample_par_degenerate_m() {
  let empty = gbm(SEED).sample_par(0);
  assert_eq!(empty.len(), 0);

  let one = gbm(SEED).sample_par(1);
  assert_eq!(one.len(), 1);

  let empty_map = gbm(SEED).sample_map(0, |p: &Array1<f64>| *p.last().unwrap());
  assert_eq!(empty_map.len(), 0);

  let one_map = gbm(SEED).sample_map(1, |p: &Array1<f64>| *p.last().unwrap());
  assert_eq!(one_map.len(), 1);
}
