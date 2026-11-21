//! A seeded `Bates` or `JumpFou` reproduces its jumps and its diffusion at any thread count.

use ndarray::Array1;
use rayon::ThreadPoolBuilder;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::euler::EulerCoefficients;
use stochastic_rs_stochastic::euler::EulerSystem;
use stochastic_rs_stochastic::jump::bates::Bates;
use stochastic_rs_stochastic::jump::jump_fou::JumpFou;
use stochastic_rs_stochastic::params::Drift;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::traits::ProcessExt;

const SEED: u64 = 42;
const N: usize = 128;
const LAMBDA: f64 = 50.0;

/// `k = 0` isolates the jump half: the drift's `-lambda*k` compensator vanishes, so a difference
/// between two `lambda` settings comes from the jump increments alone.
fn bates_with_jumps(seed: u64) -> Bates<f64, SimdNormal<f64>, Deterministic> {
  Bates::new(
    Drift::Mu(0.05),
    LAMBDA,
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
    VarianceBoundary::Floor,
    Deterministic::new(seed),
  )
}

fn bates_lambda(lambda: f64, seed: u64) -> Bates<f64, SimdNormal<f64>, Deterministic> {
  Bates::new(
    Drift::Mu(0.05),
    lambda,
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
    VarianceBoundary::Floor,
    Deterministic::new(seed),
  )
}

fn jump_fou_with_jumps(seed: u64) -> JumpFou<f64, SimdNormal<f64>, Deterministic> {
  JumpFou::new(
    0.65,
    1.5,
    0.0,
    0.2,
    LAMBDA,
    SimdNormal::new(0.0, 0.05),
    N,
    Some(0.0),
    Some(1.0),
    Deterministic::new(seed),
  )
}

fn jump_fou_lambda(lambda: f64, seed: u64) -> JumpFou<f64, SimdNormal<f64>, Deterministic> {
  JumpFou::new(
    0.65,
    1.5,
    0.0,
    0.2,
    lambda,
    SimdNormal::new(0.0, 0.05),
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
    "two identically-seeded Bates objects diverged under lambda={LAMBDA}"
  );
}

/// The price path moves when jumps are switched off, so
/// `bates_is_seed_reproducible_including_jumps` above is not a diffusion-only comparison.
#[test]
fn bates_price_path_diverges_from_zero_lambda_counterfactual() {
  let [s_jumps, _] = bates_with_jumps(SEED).sample();
  let [s_zero, _] = bates_lambda(0.0, SEED).sample();
  assert_ne!(
    bits_1d(&s_jumps),
    bits_1d(&s_zero),
    "Bates price path at lambda={LAMBDA} must differ from the lambda=0 counterfactual"
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
      "Bates sample_par diverged between 1 and 3 threads at m={m}"
    );
    assert_eq!(
      r1, r8,
      "Bates sample_par diverged between 1 and 8 threads at m={m}"
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
    "Bates sample_par produced duplicate paths at m=256"
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

/// Same "not a diffusion-only pin" proof as `Bates`'s counterpart above,
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

/// The host and the engine read one rate.
#[test]
fn lambda_is_single_sourced_at_construction() {
  let b = bates_with_jumps(SEED);
  assert_eq!(b.lambda, LAMBDA);
  assert_eq!(EulerSystem::jump_intensity(&b), Some(LAMBDA));

  let j = jump_fou_with_jumps(SEED);
  assert_eq!(j.lambda(), LAMBDA);
  assert_eq!(EulerCoefficients::jump_intensity(&j), Some(LAMBDA));
}
