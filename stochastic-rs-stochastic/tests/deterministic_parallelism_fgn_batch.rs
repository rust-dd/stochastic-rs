//! `Fgn` and `Fbm` override `sample_par` with the batched backend call, outside the trait
//! default's chunked samplers: every basis is derived before the parallel region, from their seed.

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

/// The same guarantee at a larger `m`. `Cpu::generate_batch` derives one basis per path,
/// uncapped, so there is no chunk boundary to cross here: a plain larger-`m` check.
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

/// The same larger-`m` check for `Fbm::sample_par`, which reaches the same per-path
/// `Cpu::generate_batch`.
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

/// `Fbm`'s embedded `fgn` is `Unseeded`, so the batch must read `Fbm`'s own seed: two identically
/// seeded `Fbm`s agree. Thread-count independence cannot show it: ignoring the seed passes that.
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

/// The same proof for `Fgn`, whose batch reads its own seed.
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

/// `m = 16`: no two `Fgn` paths identical; two chunks reusing one basis would collapse chunk
/// 0's paths onto chunk 1's.
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
