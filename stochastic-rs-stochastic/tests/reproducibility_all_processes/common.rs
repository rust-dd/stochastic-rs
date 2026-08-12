//! Shared infrastructure for the ten directory-grouped guard submodules
//! (`autoregressive`, `correlation`, `diffusion`, `interest`, `jump`,
//! `noise`, `process`, `rough`, `sheet`, `volatility`) — see the crate-root
//! doc comment in `../reproducibility_all_processes.rs` for the full
//! rationale, derivation of the 124-type list, and methodology notes.
//! Split out purely to keep every file under this crate's line-count limit;
//! all ten submodules plus this one compile into the single
//! `reproducibility_all_processes` test binary, so `cargo test` still runs
//! and reports all 124 checks together.

use ndarray::Array1;
use ndarray::Array2;
use num_complex::Complex;
use rayon::ThreadPoolBuilder;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::traits::ProcessExt;

pub(crate) const SEED: u64 = 42;
/// Steps per path. Small on purpose — this guard is about seed plumbing,
/// not statistics.
pub(crate) const N: usize = 24;
/// Grid size for the tempered-stable jump family (`Cgmy`/`Cts`/`KoBoL`/
/// `Rdts`/`Svcgmy`), a parameter distinct from `N`.
pub(crate) const J: usize = 8;
/// Jump intensity for the seven distribution-taking types: high enough that
/// a jump-component reproducibility bug cannot hide behind a
/// diffusion-only comparison.
pub(crate) const LAMBDA: f64 = 50.0;
/// `<= MAX_CHUNKS` (`traits/process.rs`): one path per chunk, the regime
/// that cannot expose cross-chunk correlation.
pub(crate) const M_ONE_PER_CHUNK: usize = 64;
/// `> MAX_CHUNKS`: several paths share a chunk, the regime
/// `sample_par(64)` alone cannot exercise.
pub(crate) const M_MULTI_PER_CHUNK: usize = 96;

/// A trivial `Fn1D`/`Fn2D` payload — the guard only needs *a* valid
/// callable, not a realistic curve.
pub(crate) fn fn1d_a(_t: f64) -> f64 {
  0.03
}
pub(crate) fn fn2d_a(_t: f64, _u: f64) -> f64 {
  0.01
}

/// Flattens a sampled path/output into its raw bit pattern, so `assert_eq!`
/// compares exact IEEE-754 representations rather than relying on
/// `PartialEq` semantics that could mask a `NaN`-producing divergence.
pub(crate) trait ReproBits {
  fn repro_bits(&self) -> Vec<u64>;
}

impl ReproBits for Array1<f64> {
  fn repro_bits(&self) -> Vec<u64> {
    self.iter().map(|x| x.to_bits()).collect()
  }
}

impl ReproBits for Array2<f64> {
  fn repro_bits(&self) -> Vec<u64> {
    self.iter().map(|x| x.to_bits()).collect()
  }
}

impl ReproBits for Array1<Complex<f64>> {
  fn repro_bits(&self) -> Vec<u64> {
    self
      .iter()
      .flat_map(|c| [c.re.to_bits(), c.im.to_bits()])
      .collect()
  }
}

impl ReproBits for Vec<Array1<f64>> {
  fn repro_bits(&self) -> Vec<u64> {
    self.iter().flat_map(|a| a.repro_bits()).collect()
  }
}

impl<const K: usize> ReproBits for [Array1<f64>; K] {
  fn repro_bits(&self) -> Vec<u64> {
    self.iter().flat_map(|a| a.repro_bits()).collect()
  }
}

impl<const K: usize> ReproBits for (Array1<f64>, [Array1<f64>; K]) {
  fn repro_bits(&self) -> Vec<u64> {
    let mut bits = self.0.repro_bits();
    bits.extend(self.1.repro_bits());
    bits
  }
}

pub(crate) fn pool(num_threads: usize) -> rayon::ThreadPool {
  ThreadPoolBuilder::new()
    .num_threads(num_threads)
    .build()
    .expect("failed to build rayon thread pool")
}

/// The guard's three assertions, run once per process type: (a) two fresh,
/// identically-`Deterministic`-seeded instances agree bit-for-bit on
/// `.sample()`; (b)/(c) `sample_par` agrees bit-for-bit across two rayon
/// pool sizes, both at `m <= MAX_CHUNKS` and at `m > MAX_CHUNKS`. GPU
/// backends are never reached here — every backend-generic constructor
/// called from the submodules below goes through the inherent `Cpu`-only
/// `new()`, so there is no way to build a GPU-backed instance through this
/// function at all.
pub(crate) fn check<P, F>(name: &str, make: F)
where
  F: Fn(Deterministic) -> P + Sync,
  P: ProcessExt<f64>,
  P::Output: ReproBits,
{
  let a = make(Deterministic::new(SEED)).sample().repro_bits();
  let b = make(Deterministic::new(SEED)).sample().repro_bits();
  assert_eq!(
    a, b,
    "{name}: two fresh identically-seeded instances diverged on sample()"
  );

  for &m in &[M_ONE_PER_CHUNK, M_MULTI_PER_CHUNK] {
    let run = |threads: usize| -> Vec<Vec<u64>> {
      pool(threads)
        .install(|| make(Deterministic::new(SEED)).sample_par(m))
        .iter()
        .map(|o| o.repro_bits())
        .collect()
    };
    let r1 = run(1);
    let r8 = run(8);
    assert_eq!(
      r1.len(),
      m,
      "{name}: sample_par({m}) returned {} paths, expected {m}",
      r1.len()
    );
    assert_eq!(
      r1, r8,
      "{name}: sample_par({m}) diverged between 1 and 8 rayon threads"
    );
  }
}

/// One `#[test]` per invocation, named `$test_fn`, checking `$display`
/// built by the seed-taking closure/expr `$make` — see the crate-root doc
/// comment for how the full 124-entry list was derived.
macro_rules! guard {
  ($test_fn:ident, $display:literal, $make:expr) => {
    #[test]
    fn $test_fn() {
      crate::common::check($display, $make);
    }
  };
}
pub(crate) use guard;
