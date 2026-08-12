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
//! `RoughHeston` has no jump component, so this made it **fully**
//! seed-reproducible. `Bates1996`'s jump component (`cpoisson: CompoundPoisson<T,
//! D>` structurally pinned to `Unseeded`, the same shape as
//! `Merton`/`Kou`/`LevyDiffusion`'s field of the same name) remained broken
//! independently of the `cgns` fix — closed by the
//! zero-exception-reproducibility wave's Task 2, which widened the field to
//! `CompoundPoisson<T, D, S>` and had `new()` absorb its construction (see
//! MIGRATION.md). `Bates1996` is now **fully** seed-reproducible too — this
//! file's variance-path-only tests below predate that fix and are kept
//! (still valid: the variance path never depended on jumps), plus one
//! full-output test proving the price path is reproducible now too. The
//! dedicated full-reproducibility battery (bit-identity at nonzero lambda,
//! thread-count independence at m=64/256, distinctness at m=256, all on the
//! `[s, v]` pair together) lives in `reproducibility_bates_jump_fou.rs`.

use ndarray::Array1;
use rayon::ThreadPoolBuilder;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::scalar::ScalarNormal;
use stochastic_rs_stochastic::jump::bates::Bates1996;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::fheston::RoughHeston;

const SEED: u64 = 42;
const N: usize = 128;

/// High jump intensity (mean ~50 jumps over the unit horizon at this grid,
/// via the single top-level `lambda` — see `Bates1996::lambda`'s doc for why
/// this is now the one value that reaches both the drift compensator and
/// the actual jump arrivals): makes `bates_price_path_is_seed_reproducible`
/// a real exercise of the jump component, not an unlucky all-zero-draw
/// coincidence.
fn bates(seed: u64) -> Bates1996<f64, ScalarNormal<f64>, Deterministic> {
  Bates1996::new(
    Some(0.05),
    None,
    None,
    None,
    50.0,
    0.0,
    0.04,
    1.5,
    0.3,
    -0.6,
    ScalarNormal::new(0.0, 0.05),
    N,
    Some(100.0),
    Some(0.04),
    Some(1.0),
    Some(false),
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

/// Same guarantee beyond `MAX_CHUNKS = 64`: `Bates1996` goes through
/// `ProcessExt::sample_par`'s default (not an override, unlike `Fgn`/`Fbm`),
/// so at `m = 256` several paths genuinely share one chunk's derived basis —
/// the regime `m = 64` above cannot reach.
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
    "Bates1996 variance path sample_par diverged between 1 and 8 threads at m=256"
  );
}

/// The formerly-broken half, closed by Task 2 of the
/// zero-exception-reproducibility wave (see the module doc): `cpoisson` is
/// no longer structurally pinned to `Unseeded`, so the price path — which
/// sums jump increments at every step — now agrees for two
/// identically-seeded objects too, at the same `lambda = 50` this fixture
/// already uses to make jumps fire reliably. The fuller battery (thread-count
/// independence, distinctness) lives in `reproducibility_bates_jump_fou.rs`;
/// this test just closes out the historical "still diverges" boundary this
/// file used to pin.
#[test]
fn bates_price_path_is_seed_reproducible() {
  let [sa, _] = bates(SEED).sample();
  let [sb, _] = bates(SEED).sample();
  assert_eq!(
    bits_1d(&sa),
    bits_1d(&sb),
    "Bates1996 price path diverged for two identically-seeded objects"
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

/// Same guarantee beyond `MAX_CHUNKS = 64` (see the `Bates1996` variant
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
    "RoughHeston sample_par diverged between 1 and 8 threads at m=256"
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
