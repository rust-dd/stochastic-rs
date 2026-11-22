//! `sample_par` and `sample_map` reproduce under a `Deterministic` seed at any rayon pool size,
//! which the tests pin explicitly (1, 3, 8, …): the ambient pool alone exercises no scheduling.

use ndarray::Array1;
use rayon::ThreadPoolBuilder;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::jump::cgmy::Cgmy;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::HestonPow;
use stochastic_rs_stochastic::volatility::heston::Euler;
use stochastic_rs_stochastic::volatility::heston::Heston;
use stochastic_rs_stochastic::volatility::sabr::Sabr;
use stochastic_rs_stochastic::volatility::svcgmy::Svcgmy;

const SEED: u64 = 42;

fn gbm(seed: u64) -> Gbm<f64, Deterministic> {
  Gbm::<f64, _>::new(0.05, 0.2, 32, 1.0, Some(1.0), Deterministic::new(seed))
}

/// `Heston` is the lazy class: the scheme draws per path inside the parallel closure, so its
/// sampler has to own a derived seed and never read `model.seed` there.
fn heston(seed: u64) -> Heston<f64, Euler, Deterministic> {
  Heston::new(
    1.0,
    0.04,
    2.0,
    0.04,
    0.3,
    -0.7,
    0.05,
    32,
    Some(1.0),
    HestonPow::Sqrt,
    VarianceBoundary::Absorption,
    Deterministic::new(seed),
  )
}

/// `Sabr` is the clone-snapshot class: `Deterministic::clone()` does not advance, so a sampler
/// that cloned `self.seed` would hand every chunk the same snapshot.
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

/// `Cgmy` is the tempered-stable family's base case: its per-path Gamma arrival series `P` is a
/// `Poisson` built inside `fill_path`, which has to be seeded from `self.seed` too.
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

/// `Svcgmy` builds the same per-path `Poisson` inside its fill; a `sampler()` that derives
/// correctly does not make that component seed-reproducible.
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
    0.04,
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

/// `sample_par` is bit-identical across thread-pool sizes.
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

/// The same guarantee on the lazy class (`Heston`), whose sampler owns its seed.
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

/// The same guarantee on the clone-snapshot class (`Sabr`); identical snapshots would pass this
/// and fail `sabr_sample_par_paths_are_distinct` below.
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

/// `Sabr`, `m = 64`: no two paths identical. A sampler that clones the seed gives every chunk
/// the same paths, as many distinct ones as one chunk is long.
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

/// `Sabr` beyond `MAX_CHUNKS = 64`: at `m = 256` several paths share a chunk, which `m = 64` (one
/// path per chunk) cannot show. Cloned bases one raw γ stride apart gave 78 distinct of 1000.
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

/// The same on the lazy class (`Heston`) beyond `MAX_CHUNKS`: `sampler()` derives its basis
/// once, so chunk bases stay hash-scrambled when several paths share a chunk.
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

/// Two identically seeded `Cgmy` objects agree on one `.sample()`: the Gamma arrival series `P`,
/// built inside `fill_path`, reads the seed. No chunking is involved.
#[test]
fn cgmy_sample_is_deterministic() {
  let a = bits_1d(&cgmy(SEED).sample());
  let b = bits_1d(&cgmy(SEED).sample());
  assert_eq!(a, b, "two identically-seeded Cgmy objects diverged");
}

/// The same through `sample_par` at `m = 256`: the per-path `Poisson`'s seed is derived from the
/// sampler's own chunk-decorrelated `seed`.
#[test]
fn cgmy_sample_par_is_thread_count_independent() {
  let m = 256;
  let run = |threads: usize| bits_paths(&pool(threads).install(|| cgmy(SEED).sample_par(m)));

  let r1 = run(1);
  let r8 = run(8);

  assert_eq!(r1.len(), m);
  assert_eq!(r1, r8, "Cgmy sample_par diverged between 1 and 8 threads");
}

/// `Svcgmy`: a `sampler()` that derives its seed is not enough when the per-path `Poisson`
/// inside the fill is not seeded from it.
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

/// Two identically seeded processes, the same `m`, different explicit pool sizes: identical
/// output, which also covers same-pool repeatability.
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

/// `m = 16`: no two paths are identical; two chunks consuming the same seed would collapse
/// chunk 0's paths onto chunk 1's.
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
