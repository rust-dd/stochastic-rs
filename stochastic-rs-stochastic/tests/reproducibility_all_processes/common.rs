//! The shared `check`, `guard!` and `ReproBits` of the eleven directory submodules; all compile
//! into the one `reproducibility_all_processes` test binary.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::Array3;
use num_complex::Complex;
use rayon::ThreadPoolBuilder;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::correlation::heston_stoch_corr::HestonStochCorrPaths;
use stochastic_rs_stochastic::interest::hjm::HjmPaths;
use stochastic_rs_stochastic::process::CompoundPaths;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::double_heston::DoubleHestonPaths;
use stochastic_rs_stochastic::volatility::heston_hull_white_2f::HestonHullWhite2FPaths;
use stochastic_rs_stochastic::volatility::heston2d::Heston2DPaths;
use stochastic_rs_stochastic::volatility::multifactor_heston::MultifactorHestonPaths;

pub(crate) const SEED: u64 = 42;
/// A second seed for the discrimination half of [`check`]. Any value other than [`SEED`] works:
/// `Deterministic` mixes its state with splitmix64, so adjacent seeds are not adjacent streams.
pub(crate) const OTHER_SEED: u64 = 43;
/// Steps per path. Small on purpose — this guard is about seed plumbing,
/// not statistics.
pub(crate) const N: usize = 24;
/// Grid size for the tempered-stable jump family (`Cgmy`/`Cts`/`Kobol`/
/// `Rdts`/`Svcgmy`), a parameter distinct from `N`.
pub(crate) const J: usize = 8;
/// Jump intensity for the distribution-taking types: high enough that a jump-component bug
/// cannot hide behind a diffusion-only comparison.
pub(crate) const LAMBDA: f64 = 50.0;
/// `<= MAX_CHUNKS` (`traits/process/chunks.rs`): one path per chunk, the regime
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

/// Flattens an output into its raw bits, so `assert_eq!` compares IEEE-754 representations and
/// a `NaN` cannot mask a divergence through `PartialEq`.
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

impl ReproBits for Array3<f64> {
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

impl ReproBits for HestonStochCorrPaths<f64> {
  fn repro_bits(&self) -> Vec<u64> {
    [&self.s, &self.v, &self.rho]
      .iter()
      .flat_map(|a| a.repro_bits())
      .collect()
  }
}

impl ReproBits for DoubleHestonPaths<f64> {
  fn repro_bits(&self) -> Vec<u64> {
    [&self.s, &self.v1, &self.v2]
      .iter()
      .flat_map(|a| a.repro_bits())
      .collect()
  }
}

impl ReproBits for HjmPaths<f64> {
  fn repro_bits(&self) -> Vec<u64> {
    [&self.r, &self.p, &self.f]
      .iter()
      .flat_map(|a| a.repro_bits())
      .collect()
  }
}

impl ReproBits for CompoundPaths<f64> {
  fn repro_bits(&self) -> Vec<u64> {
    [&self.arrivals, &self.cumulative, &self.jumps]
      .iter()
      .flat_map(|a| a.repro_bits())
      .collect()
  }
}

impl ReproBits for Heston2DPaths<f64> {
  fn repro_bits(&self) -> Vec<u64> {
    [&self.x1, &self.v1, &self.x2, &self.v2]
      .iter()
      .flat_map(|a| a.repro_bits())
      .collect()
  }
}

impl ReproBits for HestonHullWhite2FPaths<f64> {
  fn repro_bits(&self) -> Vec<u64> {
    [&self.s, &self.v, &self.r, &self.u, &self.integrated_rate]
      .iter()
      .flat_map(|a| a.repro_bits())
      .collect()
  }
}

impl<const K: usize> ReproBits for MultifactorHestonPaths<f64, K> {
  fn repro_bits(&self) -> Vec<u64> {
    let mut bits = self.s.repro_bits();
    bits.extend(self.v.repro_bits());
    bits
  }
}

pub(crate) fn pool(num_threads: usize) -> rayon::ThreadPool {
  ThreadPoolBuilder::new()
    .num_threads(num_threads)
    .build()
    .expect("failed to build rayon thread pool")
}

/// Per type: two fresh instances with one seed agree to the bit; [`OTHER_SEED`] differs, since a
/// dead `seed` field passes the first; `sample_par` agrees across pool sizes around `MAX_CHUNKS`.
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

  let c = make(Deterministic::new(OTHER_SEED)).sample().repro_bits();
  assert_ne!(
    a, c,
    "{name}: sample() was bit-identical under seed {SEED} and seed {OTHER_SEED} \
     — this type's seed is not reaching its sampled output. Either `self.seed` \
     (or a sub-component's seed derived from it) is never read on the sampling \
     path (the `Cir2F` class of bug), or this guard's own construction is \
     degenerate for discrimination purposes (e.g. zero volatility, an \
     intensity so low every path is empty, a deterministic limit) and needs \
     different parameters, not a weaker assertion"
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

/// One `#[test]` named `$test_fn` that checks `$display`, built by the seed-taking `$make`.
macro_rules! guard {
  ($test_fn:ident, $display:literal, $make:expr) => {
    #[test]
    fn $test_fn() {
      crate::common::check($display, $make);
    }
  };
}
pub(crate) use guard;

/// A `guard!` for a configuration its process's own guard leaves out; another name keeps the
/// `guard!` lines one per process, as `CLAUDE.md` counts them.
macro_rules! variant {
  ($test_fn:ident, $display:literal, $make:expr) => {
    #[test]
    fn $test_fn() {
      crate::common::check($display, $make);
    }
  };
}
pub(crate) use variant;
