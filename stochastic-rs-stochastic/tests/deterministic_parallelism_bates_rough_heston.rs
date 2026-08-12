//! Reproducibility tests for `Bates1996` and `RoughHeston`, whose earlier
//! "cannot be fixed this way" verdict (documented in
//! `stochastic-rs-stochastic/src/traits/process.rs` and `MIGRATION.md`) was
//! wrong: both built a correlated-Gaussian generator (`Cgns`) with the
//! literal `Unseeded` and drove it via a bare `.sample()`, which only ever
//! reads `Cgns`'s own (dead) seed field — but `Cgns::sample_impl<S2:
//! SeedExt>(&self, seed: &S2)` accepts an *external* seed, exactly how every
//! sibling `cgns`-holding type in this crate drives it. Both `sampler()`s
//! now do that.
//!
//! `RoughHeston` has no jump component, so it is now **fully**
//! seed-reproducible. `Bates1996` is not: independently of the `cgns` bug,
//! its public `cpoisson: CompoundPoisson<T, D>` field is structurally
//! pinned to `Unseeded` (the same shape as `Merton`/`Kou`/`LevyDiffusion`'s
//! field of the same name), so its jump arrivals/sizes — and therefore its
//! price path, which sums jump increments at every step — remain
//! unreproducible; only its variance path (driven solely by the now-fixed
//! `cgns`) is. `bates_price_path_jump_component_still_diverges` locks that
//! boundary in.

use ndarray::Array1;
use rayon::ThreadPoolBuilder;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::scalar::ScalarNormal;
use stochastic_rs_stochastic::jump::bates::Bates1996;
use stochastic_rs_stochastic::process::cpoisson::CompoundPoisson;
use stochastic_rs_stochastic::process::poisson::Poisson;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::fheston::RoughHeston;

const SEED: u64 = 42;
const N: usize = 128;

/// High jump intensity (mean ~50 jumps over the unit horizon at this grid):
/// makes `bates_price_path_jump_component_still_diverges` fail only if the
/// jump component were actually fixed, not by unlucky all-zero draws.
fn bates(seed: u64) -> Bates1996<f64, ScalarNormal<f64>, Deterministic> {
  Bates1996::new(
    Some(0.05),
    None,
    None,
    None,
    2.0,
    0.0,
    0.04,
    1.5,
    0.3,
    -0.6,
    N,
    Some(100.0),
    Some(0.04),
    Some(1.0),
    Some(false),
    CompoundPoisson::new(
      ScalarNormal::new(0.0, 0.05),
      Poisson::new(50.0, Some(N), Some(1.0), Unseeded),
      Unseeded,
    ),
    Deterministic::new(seed),
  )
}

fn rough_heston(seed: u64) -> RoughHeston<f64, Deterministic> {
  let mut m = RoughHeston::new(
    0.1,
    Some(0.04),
    0.04,
    1.5,
    0.3,
    None,
    None,
    Some(1.0),
    N,
    Deterministic::new(seed),
  );
  m.rho = Some(-0.6);
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

/// The fixed half: two identically-`Deterministic`-seeded `Bates1996`
/// objects must now agree on the variance path (driven solely by `cgns`).
#[test]
fn bates_variance_path_is_seed_reproducible() {
  let [_, va] = bates(SEED).sample();
  let [_, vb] = bates(SEED).sample();
  assert_eq!(
    bits_1d(&va),
    bits_1d(&vb),
    "Bates1996 variance path diverged for two identically-seeded objects"
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
    "Bates1996 variance path sample_par diverged between 1 and 3 threads"
  );
  assert_eq!(
    r1, r8,
    "Bates1996 variance path sample_par diverged between 1 and 8 threads"
  );
}

/// The still-broken half, by design: `cpoisson` stays structurally
/// `Unseeded` (see the module doc), so the price path — which sums jump
/// increments at every step — still varies run to run even under a pinned
/// seed. This pins the documented exception boundary: if this test starts
/// failing, either `Bates1996`'s jump driver became reproducible (update
/// the exception list) or the diffusion fix regressed into leaking into the
/// jump stream (a real bug).
#[test]
fn bates_price_path_jump_component_still_diverges() {
  let [sa, _] = bates(SEED).sample();
  let [sb, _] = bates(SEED).sample();
  assert_ne!(
    bits_1d(&sa),
    bits_1d(&sb),
    "expected Bates1996 price path (still Unseeded-jump-driven) to vary run to run"
  );
}

/// `RoughHeston` has no jump component, so the `cgns` fix makes it fully
/// reproducible: two identically-seeded objects must agree completely.
#[test]
fn rough_heston_sample_is_seed_reproducible() {
  let a = rough_heston(SEED).sample();
  let b = rough_heston(SEED).sample();
  assert_eq!(
    bits_2d(&a),
    bits_2d(&b),
    "two identically-seeded RoughHeston objects diverged"
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
    "RoughHeston sample_par diverged between 1 and 3 threads"
  );
  assert_eq!(
    r1, r8,
    "RoughHeston sample_par diverged between 1 and 8 threads"
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
  assert_eq!(
    keys.len(),
    m,
    "RoughHeston sample_par produced duplicate paths"
  );
}
