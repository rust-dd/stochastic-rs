//! Seeded-stream fixture: every law's stream from two seeds, pinned so that any moved draw fails.
//! Re-pin: `STREAM_SNAPSHOT_REGEN=1 cargo test -p stochastic-rs-distributions --features unstable-dual-stream-rng --test stream_snapshots -- --ignored print_constants`.

mod stream_snapshots {
  pub mod constants;
  pub mod hash;
  pub mod snapshot;
}

use std::path::Path;

use ndarray::Array2;
use ndarray::array;
use num_traits::Zero;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SimdRng;
#[cfg(feature = "unstable-dual-stream-rng")]
use stochastic_rs_core::simd_rng_dual::SimdRngDual;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::SimdKernel;
use stochastic_rs_distributions::alpha_stable::SimdAlphaStable;
use stochastic_rs_distributions::beta::SimdBeta;
use stochastic_rs_distributions::binomial::SimdBinomial;
use stochastic_rs_distributions::cauchy::SimdCauchy;
use stochastic_rs_distributions::chi_square::SimdChiSquared;
use stochastic_rs_distributions::complex::ComplexDistribution;
use stochastic_rs_distributions::dirichlet::SimdDirichlet;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::ged::SimdGed;
use stochastic_rs_distributions::generalized_hyperbolic::SimdGeneralizedHyperbolic;
use stochastic_rs_distributions::generalized_inverse_gauss::SimdGig;
use stochastic_rs_distributions::geometric::SimdGeometric;
use stochastic_rs_distributions::gev::SimdGev;
use stochastic_rs_distributions::gpd::SimdGpd;
use stochastic_rs_distributions::hypergeometric::SimdHypergeometric;
use stochastic_rs_distributions::inverse_gauss::SimdInverseGauss;
use stochastic_rs_distributions::johnson_su::SimdJohnsonSu;
use stochastic_rs_distributions::lognormal::SimdLogNormal;
use stochastic_rs_distributions::non_central_chi_squared::SimdNonCentralChiSquared;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::normal_inverse_gauss::SimdNormalInverseGauss;
use stochastic_rs_distributions::pareto::SimdPareto;
use stochastic_rs_distributions::poisson::SimdPoisson;
use stochastic_rs_distributions::skellam::SimdSkellam;
use stochastic_rs_distributions::skew_t::SimdSkewT;
use stochastic_rs_distributions::studentt::SimdStudentT;
use stochastic_rs_distributions::tempered_stable::SimdTemperedStable;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedExp;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;
use stochastic_rs_distributions::uniform::SimdUniform;
use stochastic_rs_distributions::variance_gamma::SimdVarianceGamma;
use stochastic_rs_distributions::weibull::SimdWeibull;
use stochastic_rs_distributions::wishart::SimdWishart;
use stream_snapshots::constants;
use stream_snapshots::hash::Bits;
use stream_snapshots::hash::fnv1a;
use stream_snapshots::snapshot::ScalarKind;
use stream_snapshots::snapshot::Snapshot;

const SEEDS: [u64; 2] = [42, 7];
/// Inverse of `Deterministic`'s splitmix64 increment mod 2^64, so a wrapped state difference still counts steps.
const INV_GOLDEN: u64 = 0xf1de_83e1_9937_733d;
/// `sample_matrix(64, 1024)` fans out to four workers of this many values each.
const CHUNK: usize = 16_384;

fn seeds_consumed(det: &Deterministic, seed: u64) -> u64 {
  det.current().wrapping_sub(seed).wrapping_mul(INV_GOLDEN)
}

fn chunk_firsts<T: Copy + Bits>(matrix: &Array2<T>) -> impl Iterator<Item = u64> + '_ {
  matrix.iter().step_by(CHUNK).map(|x| x.bits())
}

/// Seed budget, first eight draws, step and chunk heads and ten hashes over single, bulk, matrix and fork
/// draws; `sample_matrix` runs on the global pool, its output depends on `(m, n)` and the seed alone.
fn capture_steps<D, T>(
  seed: u64,
  make: impl Fn(&Deterministic) -> D,
  mut pop: impl FnMut(&mut D) -> T,
) -> Snapshot
where
  D: DistributionSampler<T> + Send,
  T: Copy + Zero + Bits + Send,
{
  let det = Deterministic::new(seed);
  let mut d = make(&det);
  let seeds_consumed = seeds_consumed(&det, seed);
  let single = (0..100).map(|_| pop(&mut d).bits()).collect::<Vec<_>>();
  let mut first8 = [0u64; 8];
  first8.copy_from_slice(&single[..8]);
  let mut step_heads = [0u64; 9];
  let mut chunk_heads = [0u64; 8];
  let mut hashes = [0u64; 10];
  hashes[0] = fnv1a(single.iter().copied());
  let mut buf = vec![T::zero(); 1003];
  d.fill_slice(&mut buf);
  step_heads[0] = buf[0].bits();
  hashes[1] = fnv1a(buf.iter().map(|x| x.bits()));
  // Only a fill of 8 to 15 values tells the `< 16` scalar branch from the SIMD path's scalar tail.
  let mut small = [T::zero(); 18];
  let (seven, eleven) = small.split_at_mut(7);
  d.fill_slice(seven);
  d.fill_slice(eleven);
  step_heads[1] = small[0].bits();
  step_heads[2] = small[7].bits();
  hashes[2] = fnv1a(small.iter().map(|x| x.bits()));
  // A 32-, 64- or 128-value buffer keeps 28 of the first 100 pops; 40 more refill it once, by its own size.
  let pops = (0..40).map(|_| pop(&mut d).bits()).collect::<Vec<_>>();
  // A twin without the two direct fills shares the leftover pops; the head is the first pop that differs.
  let mut unfilled = make(&Deterministic::new(seed));
  let unfilled_pops = (0..140)
    .map(|_| pop(&mut unfilled).bits())
    .collect::<Vec<_>>();
  let refill = pops
    .iter()
    .zip(&unfilled_pops[100..])
    .position(|(a, b)| a != b)
    .expect("the direct fills moved none of the 40 pops");
  step_heads[3] = pops[refill];
  hashes[3] = fnv1a(pops.iter().copied());
  let drawn = d.sample_n(1000);
  step_heads[4] = drawn[0].bits();
  hashes[4] = fnv1a(drawn.iter().map(|x| x.bits()));
  let pair = d.sample_matrix(2, 8);
  step_heads[5] = pair[[0, 0]].bits();
  hashes[5] = fnv1a(pair.iter().map(|x| x.bits()));
  let wide = d.sample_matrix(64, 1024);
  step_heads[6] = wide[[0, 0]].bits();
  for (head, bits) in chunk_heads[..4].iter_mut().zip(chunk_firsts(&wide)) {
    *head = bits;
  }
  hashes[6] = fnv1a(wide.iter().map(|x| x.bits()));
  let mut fork_buf = vec![T::zero(); 64];
  d.fork(0).fill_slice(&mut fork_buf);
  step_heads[7] = fork_buf[0].bits();
  hashes[7] = fnv1a(fork_buf.iter().map(|x| x.bits()));
  d.fork(1).fill_slice(&mut fork_buf);
  step_heads[8] = fork_buf[0].bits();
  hashes[8] = fnv1a(fork_buf.iter().map(|x| x.bits()));
  let mut twin = make(&Deterministic::new(seed));
  let _ = twin.sample_matrix(64, 1024);
  let second = twin.sample_matrix(64, 1024);
  for (head, bits) in chunk_heads[4..].iter_mut().zip(chunk_firsts(&second)) {
    *head = bits;
  }
  hashes[9] = fnv1a(second.iter().map(|x| x.bits()));
  Snapshot {
    seed,
    seeds_consumed,
    first8,
    step_heads,
    chunk_heads,
    hashes,
  }
}

/// The steps on a not yet ported law, whose `Distribution::sample` pops its own stream.
fn capture<D, T>(seed: u64, make: impl Fn(&Deterministic) -> D) -> Snapshot
where
  D: DistributionSampler<T> + Distribution<T> + Send,
  T: Copy + Zero + Bits + Send,
{
  let mut dummy = SimdRng::from_seed(1);
  capture_steps(seed, make, move |d: &mut D| d.sample(&mut dummy))
}

/// The steps on a `Seeded` stream; the constants are the ones captured on the old API.
fn capture_seeded<D, T>(seed: u64, make: impl Fn() -> D) -> Snapshot
where
  D: SimdKernel<Item = T>,
  T: Copy + Zero + Bits + Send + Sync + 'static,
{
  capture_steps(seed, |det| make().seeded(det), Seeded::sample)
}

/// Steps 1 and 4 for the laws without `fill_slice`/`fork`; `draw` yields one draw's bits.
fn capture_single<F: FnMut() -> Vec<u64>>(
  seed: u64,
  build: impl FnOnce(&Deterministic) -> F,
) -> Snapshot {
  let det = Deterministic::new(seed);
  let mut draw = build(&det);
  let seeds_consumed = seeds_consumed(&det, seed);
  let single = (0..100).flat_map(|_| draw()).collect::<Vec<_>>();
  let mut first8 = [0u64; 8];
  first8.copy_from_slice(&single[..8]);
  let tail = (0..40).flat_map(|_| draw()).collect::<Vec<_>>();
  let mut step_heads = [0u64; 9];
  step_heads[3] = tail[0];
  let mut hashes = [0u64; 10];
  hashes[0] = fnv1a(single.iter().copied());
  hashes[3] = fnv1a(tail.iter().copied());
  Snapshot {
    seed,
    seeds_consumed,
    first8,
    step_heads,
    chunk_heads: [0; 8],
    hashes,
  }
}

fn pinned(name: &str) -> &'static [Snapshot; 2] {
  constants::lookup(name)
    .unwrap_or_else(|| panic!("no constants for {name}: run the print_constants generator"))
}

fn check_close(name: &str, kind: ScalarKind, capture: impl Fn(u64) -> Snapshot) {
  for (want, seed) in pinned(name).iter().zip(SEEDS) {
    let got = capture(seed);
    assert_eq!(got.seed, want.seed);
    assert_eq!(
      got.seeds_consumed, want.seeds_consumed,
      "{name} seed {seed}: seed budget"
    );
    let parts = [
      ("first8", &got.first8[..], &want.first8[..]),
      ("step_heads", &got.step_heads[..], &want.step_heads[..]),
      ("chunk_heads", &got.chunk_heads[..], &want.chunk_heads[..]),
    ];
    for (part, got_part, want_part) in parts {
      for (i, (g, w)) in got_part.iter().zip(want_part).enumerate() {
        assert!(
          kind.close(*g, *w),
          "{name} seed {seed}: {part}[{i}] = {g:#018x}, pinned {w:#018x}"
        );
      }
    }
  }
}

fn check_exact(name: &str, capture: impl Fn(u64) -> Snapshot) {
  for (want, seed) in pinned(name).iter().zip(SEEDS) {
    let got = capture(seed);
    assert_eq!(
      got.hashes, want.hashes,
      "{name} seed {seed}: exact stream hashes"
    );
  }
}

macro_rules! snapshot_cases {
  ($( $name:ident : $kind:expr, $scalar:ty, |$det:ident| $make:expr ; )*) => {
    $(
      mod $name {
        use super::*;

        #[test]
        fn close() {
          check_close(stringify!($name), $kind, |seed| capture::<_, $scalar>(seed, |$det| $make));
        }

        #[test]
        #[cfg_attr(not(all(target_arch = "aarch64", target_os = "macos")), ignore)]
        fn exact() {
          check_exact(stringify!($name), |seed| capture::<_, $scalar>(seed, |$det| $make));
        }
      }
    )*

    fn all_cases() -> Vec<(&'static str, Box<dyn Fn(u64) -> Snapshot>)> {
      vec![$((
        stringify!($name),
        Box::new(|seed| capture::<_, $scalar>(seed, |$det| $make)) as Box<dyn Fn(u64) -> Snapshot>,
      )),*]
    }
  };
}

macro_rules! snapshot_cases_seeded {
  ($( $name:ident : $kind:expr, $scalar:ty, $make:expr ; )*) => {
    $(
      mod $name {
        use super::*;

        #[test]
        fn close() {
          check_close(stringify!($name), $kind, |seed| capture_seeded::<_, $scalar>(seed, || $make));
        }

        #[test]
        #[cfg_attr(not(all(target_arch = "aarch64", target_os = "macos")), ignore)]
        fn exact() {
          check_exact(stringify!($name), |seed| capture_seeded::<_, $scalar>(seed, || $make));
        }
      }
    )*

    fn all_seeded() -> Vec<(&'static str, Box<dyn Fn(u64) -> Snapshot>)> {
      vec![$((
        stringify!($name),
        Box::new(|seed| capture_seeded::<_, $scalar>(seed, || $make)) as Box<dyn Fn(u64) -> Snapshot>,
      )),*]
    }
  };
}

macro_rules! snapshot_singles {
  ($( $(#[$attr:meta])* $name:ident : $kind:expr, |$det:ident| $build:expr ; )*) => {
    $(
      $(#[$attr])*
      mod $name {
        use super::*;

        #[test]
        fn close() {
          check_close(stringify!($name), $kind, |seed| capture_single(seed, |$det| $build));
        }

        #[test]
        #[cfg_attr(not(all(target_arch = "aarch64", target_os = "macos")), ignore)]
        fn exact() {
          check_exact(stringify!($name), |seed| capture_single(seed, |$det| $build));
        }
      }
    )*

    fn all_singles() -> Vec<(&'static str, Box<dyn Fn(u64) -> Snapshot>)> {
      vec![$(
        $(#[$attr])*
        (
          stringify!($name),
          Box::new(|seed| capture_single(seed, |$det| $build)) as Box<dyn Fn(u64) -> Snapshot>,
        )
      ),*]
    }
  };
}

snapshot_cases_seeded! {
  normal_f64: ScalarKind::F64, f64, SimdNormal::<f64>::new(0.3, 1.7);
  normal_f32: ScalarKind::F32, f32, SimdNormal::<f32>::new(0.3, 1.7);
  exp_f64: ScalarKind::F64, f64, SimdExp::<f64>::new(1.8);
  exp_f32: ScalarKind::F32, f32, SimdExp::<f32>::new(1.8);
  uniform_f64: ScalarKind::F64, f64, SimdUniform::<f64>::new(-2.0, 3.0);
  uniform_f32: ScalarKind::F32, f32, SimdUniform::<f32>::new(-2.0, 3.0);
  uniform_unit_f64: ScalarKind::F64, f64, SimdUniform::<f64>::new(0.0, 1.0);
  uniform_unit_f32: ScalarKind::F32, f32, SimdUniform::<f32>::new(0.0, 1.0);
  gamma_f64: ScalarKind::F64, f64, SimdGamma::<f64>::new(2.5, 1.5);
  gamma_f32: ScalarKind::F32, f32, SimdGamma::<f32>::new(2.5, 1.5);
  gamma_boost_f64: ScalarKind::F64, f64, SimdGamma::<f64>::new(0.5, 2.0);
  gamma_boost_f32: ScalarKind::F32, f32, SimdGamma::<f32>::new(0.5, 2.0);
  chi_squared_f64: ScalarKind::F64, f64, SimdChiSquared::<f64>::new(6.0);
  chi_squared_f32: ScalarKind::F32, f32, SimdChiSquared::<f32>::new(6.0);
  beta_f64: ScalarKind::F64, f64, SimdBeta::<f64>::new(2.5, 4.0);
  beta_f32: ScalarKind::F32, f32, SimdBeta::<f32>::new(2.5, 4.0);
  lognormal_f64: ScalarKind::F64, f64, SimdLogNormal::<f64>::new(0.2, 0.6);
  lognormal_f32: ScalarKind::F32, f32, SimdLogNormal::<f32>::new(0.2, 0.6);
  student_t_f64: ScalarKind::F64, f64, SimdStudentT::<f64>::new(6.0);
  student_t_f32: ScalarKind::F32, f32, SimdStudentT::<f32>::new(6.0);
  cauchy_f64: ScalarKind::F64, f64, SimdCauchy::<f64>::new(1.0, 0.5);
  cauchy_f32: ScalarKind::F32, f32, SimdCauchy::<f32>::new(1.0, 0.5);
  weibull_f64: ScalarKind::F64, f64, SimdWeibull::<f64>::new(2.0, 1.5);
  weibull_f32: ScalarKind::F32, f32, SimdWeibull::<f32>::new(2.0, 1.5);
  pareto_f64: ScalarKind::F64, f64, SimdPareto::<f64>::new(1.0, 1.16);
  pareto_f32: ScalarKind::F32, f32, SimdPareto::<f32>::new(1.0, 1.16);
}

snapshot_cases! {
  alpha_stable_f64: ScalarKind::F64, f64, |det| SimdAlphaStable::<f64>::new(1.7, 0.3, 1.0, 0.0, det);
  alpha_stable_f32: ScalarKind::F32, f32, |det| SimdAlphaStable::<f32>::new(1.7, 0.3, 1.0, 0.0, det);
  alpha_stable_cauchy_f64: ScalarKind::F64, f64, |det| SimdAlphaStable::<f64>::new(1.0, 0.3, 1.0, 0.0, det);
  alpha_stable_cauchy_f32: ScalarKind::F32, f32, |det| SimdAlphaStable::<f32>::new(1.0, 0.3, 1.0, 0.0, det);
  alpha_stable_gauss_f64: ScalarKind::F64, f64, |det| SimdAlphaStable::<f64>::new(2.0, 0.0, 1.0, 0.0, det);
  alpha_stable_gauss_f32: ScalarKind::F32, f32, |det| SimdAlphaStable::<f32>::new(2.0, 0.0, 1.0, 0.0, det);
  tempered_stable_f64: ScalarKind::F64, f64, |det| SimdTemperedStable::<f64>::new(0.6, 2.0, 1.5, det);
  tempered_stable_f32: ScalarKind::F32, f32, |det| SimdTemperedStable::<f32>::new(0.6, 2.0, 1.5, det);
  nig_f64: ScalarKind::F64, f64, |det| SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 1.0, 0.1, det);
  nig_f32: ScalarKind::F32, f32, |det| SimdNormalInverseGauss::<f32>::new(2.0, 0.5, 1.0, 0.1, det);
  variance_gamma_f64: ScalarKind::F64, f64, |det| SimdVarianceGamma::<f64>::new(0.2, 0.5, -0.1, 0.05, det);
  variance_gamma_f32: ScalarKind::F32, f32, |det| SimdVarianceGamma::<f32>::new(0.2, 0.5, -0.1, 0.05, det);
  generalized_hyperbolic_f64: ScalarKind::F64, f64, |det| SimdGeneralizedHyperbolic::<f64>::new(1.0, 2.0, 0.5, 1.5, -0.2, det);
  generalized_hyperbolic_f32: ScalarKind::F32, f32, |det| SimdGeneralizedHyperbolic::<f32>::new(1.0, 2.0, 0.5, 1.5, -0.2, det);
  gig_a_f64: ScalarKind::F64, f64, |det| SimdGig::<f64>::new(0.3, 2.0, 0.5, det);
  gig_a_f32: ScalarKind::F32, f32, |det| SimdGig::<f32>::new(0.3, 2.0, 0.5, det);
  gig_b_f64: ScalarKind::F64, f64, |det| SimdGig::<f64>::new(0.2, 0.01, 1.0, det);
  gig_b_f32: ScalarKind::F32, f32, |det| SimdGig::<f32>::new(0.2, 0.01, 1.0, det);
  gig_c_f64: ScalarKind::F64, f64, |det| SimdGig::<f64>::new(3.0, 1.0, 1.0, det);
  gig_c_f32: ScalarKind::F32, f32, |det| SimdGig::<f32>::new(3.0, 1.0, 1.0, det);
  gig_negative_f64: ScalarKind::F64, f64, |det| SimdGig::<f64>::new(-0.5, 2.0, 0.5, det);
  gig_negative_f32: ScalarKind::F32, f32, |det| SimdGig::<f32>::new(-0.5, 2.0, 0.5, det);
  inverse_gauss_f64: ScalarKind::F64, f64, |det| SimdInverseGauss::<f64>::new(1.5, 3.0, det);
  inverse_gauss_f32: ScalarKind::F32, f32, |det| SimdInverseGauss::<f32>::new(1.5, 3.0, det);
  skew_t_f64: ScalarKind::F64, f64, |det| SimdSkewT::<f64>::new(5.0, -0.3, det);
  skew_t_f32: ScalarKind::F32, f32, |det| SimdSkewT::<f32>::new(5.0, -0.3, det);
  johnson_su_f64: ScalarKind::F64, f64, |det| SimdJohnsonSu::<f64>::new(-0.5, 1.5, 0.2, 2.0, det);
  johnson_su_f32: ScalarKind::F32, f32, |det| SimdJohnsonSu::<f32>::new(-0.5, 1.5, 0.2, 2.0, det);
  ged_f64: ScalarKind::F64, f64, |det| SimdGed::<f64>::new(0.0, 1.0, 1.5, det);
  ged_f32: ScalarKind::F32, f32, |det| SimdGed::<f32>::new(0.0, 1.0, 1.5, det);
  gev_frechet_f64: ScalarKind::F64, f64, |det| SimdGev::<f64>::new(0.0, 1.0, 0.3, det);
  gev_frechet_f32: ScalarKind::F32, f32, |det| SimdGev::<f32>::new(0.0, 1.0, 0.3, det);
  gev_gumbel_f64: ScalarKind::F64, f64, |det| SimdGev::<f64>::new(0.0, 1.0, 0.0, det);
  gev_gumbel_f32: ScalarKind::F32, f32, |det| SimdGev::<f32>::new(0.0, 1.0, 0.0, det);
  gev_weibull_f64: ScalarKind::F64, f64, |det| SimdGev::<f64>::new(0.0, 1.0, -0.3, det);
  gev_weibull_f32: ScalarKind::F32, f32, |det| SimdGev::<f32>::new(0.0, 1.0, -0.3, det);
  gpd_frechet_f64: ScalarKind::F64, f64, |det| SimdGpd::<f64>::new(0.0, 1.0, 0.3, det);
  gpd_frechet_f32: ScalarKind::F32, f32, |det| SimdGpd::<f32>::new(0.0, 1.0, 0.3, det);
  gpd_exp_f64: ScalarKind::F64, f64, |det| SimdGpd::<f64>::new(0.0, 1.0, 0.0, det);
  gpd_exp_f32: ScalarKind::F32, f32, |det| SimdGpd::<f32>::new(0.0, 1.0, 0.0, det);
  gpd_bounded_f64: ScalarKind::F64, f64, |det| SimdGpd::<f64>::new(0.0, 1.0, -0.3, det);
  gpd_bounded_f32: ScalarKind::F32, f32, |det| SimdGpd::<f32>::new(0.0, 1.0, -0.3, det);
  poisson_u32: ScalarKind::Int, u32, |det| SimdPoisson::<u32>::new(12.0, det);
  poisson_u64: ScalarKind::Int, u64, |det| SimdPoisson::<u64>::new(12.0, det);
  poisson_i64: ScalarKind::Int, i64, |det| SimdPoisson::<i64>::new(12.0, det);
  poisson_large_u64: ScalarKind::Int, u64, |det| SimdPoisson::<u64>::new(800.0, det);
  binomial_btrs_u32: ScalarKind::Int, u32, |det| SimdBinomial::<u32>::new(60, 0.4, det);
  binomial_waiting_u32: ScalarKind::Int, u32, |det| SimdBinomial::<u32>::new(15, 0.3, det);
  binomial_btrs_flip_u32: ScalarKind::Int, u32, |det| SimdBinomial::<u32>::new(60, 0.7, det);
  binomial_waiting_flip_u32: ScalarKind::Int, u32, |det| SimdBinomial::<u32>::new(15, 0.8, det);
  geometric_u64: ScalarKind::Int, u64, |det| SimdGeometric::<u64>::new(0.15, det);
  hypergeometric_u32: ScalarKind::Int, u32, |det| SimdHypergeometric::<u32>::new(60, 25, 20, det);
}

snapshot_singles! {
  skellam: ScalarKind::Int, |det| {
    let d = SimdSkellam::<SimdRng>::new(9.0, 5.0, det);
    move || vec![d.sample_fast().bits()]
  };
  dirichlet_f64: ScalarKind::F64, |det| {
    let d = SimdDirichlet::<f64>::new(vec![1.0, 2.0, 3.0], det);
    let mut out = vec![0.0f64; 3];
    move || {
      d.sample_into(&mut out);
      out.iter().map(|x| x.bits()).collect()
    }
  };
  dirichlet_f32: ScalarKind::F32, |det| {
    let d = SimdDirichlet::<f32>::new(vec![1.0, 2.0, 3.0], det);
    let mut out = vec![0.0f32; 3];
    move || {
      d.sample_into(&mut out);
      out.iter().map(|x| x.bits()).collect()
    }
  };
  wishart_f64: ScalarKind::F64, |det| {
    let d = SimdWishart::<f64>::new(5.0, array![[1.0, 0.3], [0.3, 2.0]], det);
    move || d.sample_fast().iter().map(|x| x.bits()).collect()
  };
  ncx2_f64: ScalarKind::F64, |det| {
    let d = SimdNonCentralChiSquared::<f64>::new(3.0, det);
    move || vec![d.sample_ncp(2.5).bits()]
  };
  ncx2_f32: ScalarKind::F32, |det| {
    let d = SimdNonCentralChiSquared::<f32>::new(3.0, det);
    move || vec![d.sample_ncp(2.5).bits()]
  };
  ncx2_mixture_f64: ScalarKind::F64, |det| {
    let d = SimdNonCentralChiSquared::<f64>::new(0.3, det);
    move || vec![d.sample_ncp(2.0).bits()]
  };
  truncated_normal_f64: ScalarKind::F64, |det| {
    let d = SimdTruncatedNormal::<f64>::new(0.0, 1.0, -1.0, 2.0, det);
    move || vec![d.sample_fast().bits()]
  };
  truncated_normal_f32: ScalarKind::F32, |det| {
    let d = SimdTruncatedNormal::<f32>::new(0.0, 1.0, -1.0, 2.0, det);
    move || vec![d.sample_fast().bits()]
  };
  truncated_normal_exp_tail_f64: ScalarKind::F64, |det| {
    let d = SimdTruncatedNormal::<f64>::new(0.0, 1.0, 3.0, 6.0, det);
    move || vec![d.sample_fast().bits()]
  };
  truncated_normal_uniform_tail_f64: ScalarKind::F64, |det| {
    let d = SimdTruncatedNormal::<f64>::new(0.0, 1.0, 3.0, 3.4, det);
    move || vec![d.sample_fast().bits()]
  };
  truncated_normal_mirrored_f64: ScalarKind::F64, |det| {
    let d = SimdTruncatedNormal::<f64>::new(0.0, 1.0, -6.0, -3.0, det);
    move || vec![d.sample_fast().bits()]
  };
  truncated_normal_inverse_cdf_f64: ScalarKind::F64, |det| {
    let d = SimdTruncatedNormal::<f64>::new(0.0, 1.0, -0.05, 0.05, det);
    move || vec![d.sample_fast().bits()]
  };
  truncated_exp_f64: ScalarKind::F64, |det| {
    let d = SimdTruncatedExp::<f64>::new(2.0, 0.0, 1.5, det);
    move || vec![d.sample_fast().bits()]
  };
  truncated_exp_f32: ScalarKind::F32, |det| {
    let d = SimdTruncatedExp::<f32>::new(2.0, 0.0, 1.5, det);
    move || vec![d.sample_fast().bits()]
  };
  truncated_beta_f64: ScalarKind::F64, |det| {
    let d = SimdTruncatedBeta::<f64>::new(2.0, 2.0, 0.2, 0.8, det);
    move || vec![d.sample_fast().bits()]
  };
  truncated_beta_f32: ScalarKind::F32, |det| {
    let d = SimdTruncatedBeta::<f32>::new(2.0, 2.0, 0.2, 0.8, det);
    move || vec![d.sample_fast().bits()]
  };
  truncated_gamma_f64: ScalarKind::F64, |det| {
    let d = SimdTruncatedGamma::<f64>::new(2.0, 1.0, 1.0, 5.0, det);
    move || vec![d.sample_fast().bits()]
  };
  truncated_gamma_f32: ScalarKind::F32, |det| {
    let d = SimdTruncatedGamma::<f32>::new(2.0, 1.0, 1.0, 5.0, det);
    move || vec![d.sample_fast().bits()]
  };
  complex_normal_f64: ScalarKind::F64, |det| {
    let mut s = ComplexDistribution::new(
      SimdNormal::<f64>::new(0.0, 1.0),
      SimdNormal::<f64>::new(0.5, 2.0),
    )
    .seeded(det);
    move || {
      let z = s.sample();
      vec![z.re.bits(), z.im.bits()]
    }
  };
  #[cfg(feature = "unstable-dual-stream-rng")]
  normal_dual_f64: ScalarKind::F64, |det| {
    let mut d = Seeded::<SimdNormal<f64>, SimdRngDual>::new(SimdNormal::new(0.3, 1.7), det);
    move || vec![d.sample().bits()]
  };
  #[cfg(feature = "unstable-dual-stream-rng")]
  normal_dual_f32: ScalarKind::F32, |det| {
    let mut d = Seeded::<SimdNormal<f32>, SimdRngDual>::new(SimdNormal::new(0.3, 1.7), det);
    move || vec![d.sample().bits()]
  };
  #[cfg(feature = "unstable-dual-stream-rng")]
  exp_dual_f64: ScalarKind::F64, |det| {
    let mut d = Seeded::<SimdExp<f64>, SimdRngDual>::new(SimdExp::new(1.8), det);
    move || vec![d.sample().bits()]
  };
  #[cfg(feature = "unstable-dual-stream-rng")]
  exp_dual_f32: ScalarKind::F32, |det| {
    let mut d = Seeded::<SimdExp<f32>, SimdRngDual>::new(SimdExp::new(1.8), det);
    move || vec![d.sample().bits()]
  };
}

/// Writes `constants.rs` under `STREAM_SNAPSHOT_REGEN=1`, only for a re-pin with a `### Numerical changes` entry.
#[test]
#[ignore = "re-pin generator, gated on STREAM_SNAPSHOT_REGEN=1 (command in the module doc)"]
fn print_constants() {
  if std::env::var("STREAM_SNAPSHOT_REGEN").as_deref() != Ok("1") {
    println!("print_constants: set STREAM_SNAPSHOT_REGEN=1 to rewrite constants.rs");
    return;
  }
  if cfg!(not(feature = "unstable-dual-stream-rng")) {
    panic!(
      "regenerate with --features unstable-dual-stream-rng so the dual-stream cases keep their constants"
    );
  }
  let mut text = String::from(
    "//! Generated by `print_constants` in `../stream_snapshots.rs`; rerun it only for an intentional re-pin.\n\nuse super::snapshot::Snapshot;\n\n",
  );
  let mut names = Vec::new();
  for (name, capture) in all_cases()
    .into_iter()
    .chain(all_seeded())
    .chain(all_singles())
  {
    let snapshots = SEEDS.map(capture);
    text.push_str(&format!(
      "pub const {}: [Snapshot; 2] = {snapshots:#?};\n\n",
      name.to_uppercase()
    ));
    names.push(name);
  }
  text.push_str("pub fn lookup(name: &str) -> Option<&'static [Snapshot; 2]> {\n  match name {\n");
  for name in &names {
    text.push_str(&format!(
      "    \"{name}\" => Some(&{}),\n",
      name.to_uppercase()
    ));
  }
  text.push_str("    _ => None,\n  }\n}\n");
  let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/stream_snapshots/constants.rs");
  std::fs::write(&path, text).unwrap();
  println!("wrote {} ({} cases)", path.display(), names.len());
}
