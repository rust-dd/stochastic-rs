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
//! in addition to `Gbm`.
//!
//! The same review found the fix was a no-op for processes whose
//! `sampler()` reads the seed *lazily* per path from inside the returned
//! sampler (`Heston`, and 9 more) — advancing the seed sequentially before
//! any chunk runs cannot help a process that never advances anything at
//! `sampler()` construction; each chunk's sampler shares live access to the
//! same atomic and races on it during the parallel region itself.
//! `heston_sample_par_is_thread_count_independent` below covers that class,
//! fixed by rewriting the sampler to own a seed rather than reading the
//! process's directly.
//!
//! A third review found a distinct bug class neither the `seed.clone()` grep
//! nor a `sampler()`-body sweep can see: `Cgmy`, `KoBoL`, `Cts`, `Rdts` and
//! `Svcgmy` each hard-wire `Unseeded` on a `Poisson` built *inside*
//! `fill_path`/`fill_paths` (an arrival-time series reused as Γ_j), so that
//! one component of the path was never reachable from `self.seed` at all —
//! not a chunking defect, a plain missed wire. `cgmy_sample_is_deterministic`
//! / `svcgmy_sample_is_deterministic` below cover the two the review asked
//! for by name; `KoBoL`/`Cts`/`Rdts` got the identical one-line fix without
//! their own dedicated tests here (see the source for the shared pattern).

use ndarray::Array1;
use rayon::ThreadPoolBuilder;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::jump::cgmy::Cgmy;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::HestonPow;
use stochastic_rs_stochastic::volatility::heston::Heston;
use stochastic_rs_stochastic::volatility::sabr::Sabr;
use stochastic_rs_stochastic::volatility::svcgmy::Svcgmy;

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

/// `Heston` is "lazy": pre-fix, `sampler()` did nothing and `Sch::simulate`
/// read `model.seed.derive()` fresh per path, from inside the (potentially
/// parallel) closure — advancing `self.seed` at construction alone cannot
/// fix a process that never advances anything at construction.
fn heston(seed: u64) -> Heston<f64, Deterministic> {
  Heston::<f64, _>::new(
    Some(1.0),
    Some(0.04),
    2.0,
    0.04,
    0.3,
    -0.7,
    0.05,
    32,
    Some(1.0),
    HestonPow::Sqrt,
    None,
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

/// `Cgmy` is the tempered-stable jump family's base case: its per-path
/// Gamma-arrival-time series `P` was built via `Poisson::new(..., Unseeded)`
/// inside `fill_path`, entirely bypassing `self.seed`.
fn cgmy(seed: u64) -> Cgmy<f64, Deterministic> {
  Cgmy::<f64, _>::new(
    1.0,
    3.0,
    3.0,
    0.5,
    32,
    8,
    Some(0.0),
    Some(1.0),
    Deterministic::new(seed),
  )
}

/// `Svcgmy` had the identical hard-wired-`Unseeded` `Poisson` bug, but is
/// the sharper case: its `sampler()` was already fixed to `derive()` (not
/// `clone()`) in the cross-chunk-correlation round, so it was covered by
/// that round's "Guarantee, corrected" claim while still not being
/// seed-reproducible at all, via this one line.
fn svcgmy(seed: u64) -> Svcgmy<f64, Deterministic> {
  Svcgmy::<f64, _>::new(
    1.0,
    1.0,
    0.5,
    2.0,
    0.04,
    0.2,
    0.0,
    32,
    8,
    Some(0.0),
    Some(0.04),
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

/// Same guarantee, on the "lazy" class (`Heston`): pre-fix, this was a
/// no-op — the fix has to rewrite `sampler()` to own the seed, not merely
/// pre-build samplers sequentially.
#[test]
fn heston_sample_par_is_thread_count_independent() {
  let m = 64;
  let run = |threads: usize| {
    pool(threads)
      .install(|| heston(SEED).sample_par(m))
      .iter()
      .map(bits_2d)
      .collect::<Vec<_>>()
  };

  let r1 = run(1);
  let r3 = run(3);
  let r8 = run(8);

  assert_eq!(r1.len(), m);
  assert_eq!(r1, r3, "Heston sample_par diverged between 1 and 3 threads");
  assert_eq!(r1, r8, "Heston sample_par diverged between 1 and 8 threads");
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
/// test that catches the clone-snapshot regression specifically. Under the
/// very first fix (the `m.div_ceil(8)` chunking rule — `MAX_CHUNKS` did not
/// exist at that point in the fix's history), every chunk cloned the same
/// unchanged seed, so `sample_par(m)` degenerated to only as many distinct
/// paths as one chunk's own LENGTH (8, under that rule — a path-count, not
/// a chunk-count), each repeated across every chunk.
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

/// `Sabr` (clone-snapshot) beyond `MAX_CHUNKS = 64`: at `m = 256`,
/// `chunk_count` caps at 64 chunks, so several chunks carry more than one
/// path each — the regime the `m = 64` test above cannot exercise (there,
/// `chunk_count(64) == 64` gives exactly one path per chunk, which cannot
/// expose cross-chunk correlation). Before `sampler()` derived — rather
/// than cloned — its chunk basis, adjacent chunks' bases were a raw,
/// unmixed γ stride apart, so `sample_par(1000)` measured only 78 of 1000
/// paths distinct; deriving gives every chunk a hash-scrambled,
/// uncorrelated basis regardless of how many paths share it.
#[test]
fn sabr_sample_par_paths_are_distinct_beyond_max_chunks() {
  let m = 256;
  let paths = sabr(SEED).sample_par(m);
  assert_eq!(paths.len(), m);
  let keys = paths
    .iter()
    .map(bits_2d)
    .collect::<std::collections::HashSet<_>>();
  assert_eq!(
    keys.len(),
    m,
    "Sabr sample_par produced duplicate paths beyond MAX_CHUNKS"
  );
}

/// Same guarantee on the "lazy-rewritten" class (`Heston`) beyond
/// `MAX_CHUNKS`: `Heston::sampler()` now derives its own basis once (rather
/// than the process's `self.seed` being read fresh per path), so chunk
/// bases stay hash-scrambled relative to each other even when `m` forces
/// multiple paths onto the same chunk.
#[test]
fn heston_sample_par_paths_are_distinct_beyond_max_chunks() {
  let m = 256;
  let paths = heston(SEED).sample_par(m);
  assert_eq!(paths.len(), m);
  let keys = paths
    .iter()
    .map(bits_2d)
    .collect::<std::collections::HashSet<_>>();
  assert_eq!(
    keys.len(),
    m,
    "Heston sample_par produced duplicate paths beyond MAX_CHUNKS"
  );
}

/// `Cgmy`'s per-path Gamma-arrival-time series `P` was hard-wired to
/// `Unseeded` inside `fill_path` — invisible to both the `seed.clone()` grep
/// and the `sampler()`-body sweep that found every other bug in this suite,
/// since `sampler()` itself is correctly seeded and the defect is a plain
/// object *inside* the per-path method. Before the fix, two identically-
/// `Deterministic`-seeded `Cgmy` objects disagreed on a single `.sample()`
/// call — not a chunking defect, `sample_par`/`sample_map` are not even
/// involved here.
#[test]
fn cgmy_sample_is_deterministic() {
  let a = bits_1d(&cgmy(SEED).sample());
  let b = bits_1d(&cgmy(SEED).sample());
  assert_eq!(a, b, "two identically-seeded Cgmy objects diverged");
}

/// Same bug as `cgmy_sample_is_deterministic`, exercised through
/// `sample_par` at `m = 256`: fixed by deriving the per-path `Poisson`'s
/// seed from the sampler's own (already chunk-decorrelated) `seed` field
/// instead of hard-wiring `Unseeded`.
#[test]
fn cgmy_sample_par_is_thread_count_independent() {
  let m = 256;
  let run = |threads: usize| bits_paths(&pool(threads).install(|| cgmy(SEED).sample_par(m)));

  let r1 = run(1);
  let r8 = run(8);

  assert_eq!(r1.len(), m);
  assert_eq!(r1, r8, "Cgmy sample_par diverged between 1 and 8 threads");
}

/// `Svcgmy` — the review's named "sharp" case: this file's own
/// cross-chunk-correlation fix already converted its `sampler()` from
/// `clone()` to `derive()`, so it was covered by that round's "Guarantee,
/// corrected" claim while still not being seed-reproducible at all, via the
/// identical hard-wired-`Unseeded` per-path `Poisson` as `Cgmy`.
#[test]
fn svcgmy_sample_is_deterministic() {
  let a = svcgmy(SEED).sample();
  let b = svcgmy(SEED).sample();
  assert_eq!(
    bits_2d(&a),
    bits_2d(&b),
    "two identically-seeded Svcgmy objects diverged"
  );
}

/// Same bug as `svcgmy_sample_is_deterministic`, through `sample_par` at
/// `m = 256`.
#[test]
fn svcgmy_sample_par_is_thread_count_independent() {
  let m = 256;
  let run = |threads: usize| {
    pool(threads)
      .install(|| svcgmy(SEED).sample_par(m))
      .iter()
      .map(bits_2d)
      .collect::<Vec<_>>()
  };

  let r1 = run(1);
  let r8 = run(8);

  assert_eq!(r1.len(), m);
  assert_eq!(r1, r8, "Svcgmy sample_par diverged between 1 and 8 threads");
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
