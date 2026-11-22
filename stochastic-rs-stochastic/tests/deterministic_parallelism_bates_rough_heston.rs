//! `Bates` and `Fheston` draw every shock from their seed, so identically seeded processes agree at
//! any thread count; `reproducibility_bates_jump_fou.rs` holds `Bates`'s full jump battery.

use ndarray::Array1;
use rayon::ThreadPoolBuilder;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::jump::bates::Bates;
use stochastic_rs_stochastic::params::Drift;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::fheston::Fheston;

const SEED: u64 = 42;
const N: usize = 128;

/// λ = 50 over the unit horizon, so comparing price paths exercises the jumps.
fn bates(seed: u64) -> Bates<f64, SimdNormal<f64>, Deterministic> {
  Bates::new(
    Drift::Mu(0.05),
    50.0,
    0.0,
    0.04,
    1.5,
    0.3,
    -0.6,
    SimdNormal::new(0.0, 0.05),
    N,
    100.0,
    0.04,
    Some(1.0),
    VarianceBoundary::Absorption,
    Deterministic::new(seed),
  )
}

fn rough_heston(seed: u64) -> Fheston<f64, Deterministic> {
  let mut m = Fheston::new(
    0.1,
    0.04,
    0.04,
    1.5,
    0.3,
    None,
    None,
    Some(1.0),
    N,
    Deterministic::new(seed),
  );
  m.rho = -0.6;
  m
}

fn bits_1d(path: &Array1<f64>) -> Vec<u64> {
  path.iter().map(|x| x.to_bits()).collect()
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

/// Two identically seeded `Bates` agree on the variance path, which `cgns` drives alone.
#[test]
fn bates_variance_path_is_seed_reproducible() {
  let [_, va] = bates(SEED).sample();
  let [_, vb] = bates(SEED).sample();
  assert_eq!(
    bits_1d(&va),
    bits_1d(&vb),
    "Bates variance path diverged for two identically-seeded objects"
  );
}

/// Same guarantee through `sample_par`, across thread-pool sizes.
#[test]
fn bates_variance_path_sample_par_is_thread_count_independent() {
  let m = 64;
  let run = |threads: usize| {
    pool(threads)
      .install(|| bates(SEED).sample_par(m))
      .iter()
      .map(|[_, v]| bits_1d(v))
      .collect::<Vec<_>>()
  };

  let r1 = run(1);
  let r3 = run(3);
  let r8 = run(8);

  assert_eq!(r1.len(), m);
  assert_eq!(
    r1, r3,
    "Bates variance path sample_par diverged between 1 and 3 threads"
  );
  assert_eq!(
    r1, r8,
    "Bates variance path sample_par diverged between 1 and 8 threads"
  );
}

/// The same guarantee beyond `MAX_CHUNKS = 64`: `Bates` takes `ProcessExt::sample_par`'s default,
/// so at `m = 256` several paths share one chunk's derived basis, which `m = 64` cannot reach.
#[test]
fn bates_variance_path_sample_par_is_thread_count_independent_beyond_max_chunks() {
  let m = 256;
  let run = |threads: usize| {
    pool(threads)
      .install(|| bates(SEED).sample_par(m))
      .iter()
      .map(|[_, v]| bits_1d(v))
      .collect::<Vec<_>>()
  };

  let r1 = run(1);
  let r8 = run(8);

  assert_eq!(r1.len(), m);
  assert_eq!(
    r1, r8,
    "Bates variance path sample_par diverged between 1 and 8 threads at m=256"
  );
}

/// The price path sums a jump increment at every step, so two identically seeded `Bates` agree on
/// it only if the jumps draw from the seed too.
#[test]
fn bates_price_path_is_seed_reproducible() {
  let [sa, _] = bates(SEED).sample();
  let [sb, _] = bates(SEED).sample();
  assert_eq!(
    bits_1d(&sa),
    bits_1d(&sb),
    "Bates price path diverged for two identically-seeded objects"
  );
}

/// `Fheston` has no jumps, so two identically seeded processes agree on both paths.
#[test]
fn rough_heston_sample_is_seed_reproducible() {
  let a = rough_heston(SEED).sample();
  let b = rough_heston(SEED).sample();
  assert_eq!(
    bits_2d(&a),
    bits_2d(&b),
    "two identically-seeded Fheston objects diverged"
  );
}

#[test]
fn rough_heston_sample_par_is_thread_count_independent() {
  let m = 64;
  let run = |threads: usize| {
    pool(threads)
      .install(|| rough_heston(SEED).sample_par(m))
      .iter()
      .map(bits_2d)
      .collect::<Vec<_>>()
  };

  let r1 = run(1);
  let r3 = run(3);
  let r8 = run(8);

  assert_eq!(r1.len(), m);
  assert_eq!(
    r1, r3,
    "Fheston sample_par diverged between 1 and 3 threads"
  );
  assert_eq!(
    r1, r8,
    "Fheston sample_par diverged between 1 and 8 threads"
  );
}

/// Same guarantee beyond `MAX_CHUNKS = 64` (see the `Bates` variant
/// above for why this regime matters).
#[test]
fn rough_heston_sample_par_is_thread_count_independent_beyond_max_chunks() {
  let m = 256;
  let run = |threads: usize| {
    pool(threads)
      .install(|| rough_heston(SEED).sample_par(m))
      .iter()
      .map(bits_2d)
      .collect::<Vec<_>>()
  };

  let r1 = run(1);
  let r8 = run(8);

  assert_eq!(r1.len(), m);
  assert_eq!(
    r1, r8,
    "Fheston sample_par diverged between 1 and 8 threads at m=256"
  );
}

#[test]
fn rough_heston_sample_par_paths_are_distinct() {
  let m = 16;
  let paths = rough_heston(SEED).sample_par(m);
  assert_eq!(paths.len(), m);
  let keys = paths
    .iter()
    .map(bits_2d)
    .collect::<std::collections::HashSet<_>>();
  assert_eq!(keys.len(), m, "Fheston sample_par produced duplicate paths");
}
