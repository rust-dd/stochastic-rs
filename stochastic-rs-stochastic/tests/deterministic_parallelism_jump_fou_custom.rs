//! `JumpFouCustom` draws its fGN from a child of its seed, `fgn` lending only the FFT plan, and its
//! jumps from the seed too, so identically seeded processes agree at any thread count.

use ndarray::Array1;
use rayon::ThreadPoolBuilder;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_stochastic::jump::jump_fou_custom::JumpFouCustom;
use stochastic_rs_stochastic::traits::ProcessExt;

const SEED: u64 = 42;
const N: usize = 128;

fn jump_fou_custom(seed: u64) -> JumpFouCustom<f64, SimdExp<f64>, SimdExp<f64>, Deterministic> {
  JumpFouCustom::new(
    0.65,
    1.5,
    0.0,
    0.2,
    N,
    Some(0.0),
    Some(1.0),
    SimdExp::new(20.0),
    SimdExp::new(5.0),
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
fn jump_fou_custom_sample_is_seed_reproducible() {
  let a = jump_fou_custom(SEED).sample();
  let b = jump_fou_custom(SEED).sample();
  assert_eq!(
    bits_1d(&a),
    bits_1d(&b),
    "two identically-seeded JumpFouCustom objects diverged"
  );
}

#[test]
fn jump_fou_custom_sample_par_is_thread_count_independent() {
  let m = 64;
  let run =
    |threads: usize| bits_paths(&pool(threads).install(|| jump_fou_custom(SEED).sample_par(m)));

  let r1 = run(1);
  let r3 = run(3);
  let r8 = run(8);

  assert_eq!(r1.len(), m);
  assert_eq!(
    r1, r3,
    "JumpFouCustom sample_par diverged between 1 and 3 threads"
  );
  assert_eq!(
    r1, r8,
    "JumpFouCustom sample_par diverged between 1 and 8 threads"
  );
}

/// Same guarantee beyond `MAX_CHUNKS = 64`: `JumpFouCustom` goes through
/// `ProcessExt::sample_par`'s default, so at `m = 256` several paths share
/// one chunk's derived basis — the regime `m = 64` above cannot reach.
#[test]
fn jump_fou_custom_sample_par_is_thread_count_independent_beyond_max_chunks() {
  let m = 256;
  let run =
    |threads: usize| bits_paths(&pool(threads).install(|| jump_fou_custom(SEED).sample_par(m)));

  let r1 = run(1);
  let r8 = run(8);

  assert_eq!(r1.len(), m);
  assert_eq!(
    r1, r8,
    "JumpFouCustom sample_par diverged between 1 and 8 threads at m=256"
  );
}

#[test]
fn jump_fou_custom_sample_par_paths_are_distinct() {
  let m = 16;
  let paths = jump_fou_custom(SEED).sample_par(m);
  assert_eq!(paths.len(), m);
  let keys = paths
    .iter()
    .map(bits_1d)
    .collect::<std::collections::HashSet<_>>();
  assert_eq!(
    keys.len(),
    m,
    "JumpFouCustom sample_par produced duplicate paths"
  );
}
