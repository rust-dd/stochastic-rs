//! Golden stream tests for the sampler-v3 refactor.
//!
//! Captured on the pre-refactor tree: with [`Deterministic`] seeding these
//! values must survive the `ProcessExt` → `PathSampler` migration, anchoring
//! that the derived RNG streams are preserved. They are compared with a small
//! tolerance rather than bit-for-bit: FFT / `powf`-heavy paths (FGN) round
//! differently in their low bits across architectures (x86 vs ARM), so a
//! pinned bit pattern is not portable. Exact reproduction of the refactor
//! itself is covered bit-for-bit, machine-independently, by
//! [`sampler_first_path_matches_sample`].
//!
//! `golden_merton_streams` below is the first golden covering a jump chain:
//! before the zero-exception-reproducibility wave's Task 1, `Merton` hard-
//! wired its inner `CompoundPoisson<T, D>` to `Unseeded`, so its jump chain
//! was not bit-reproducible and could not be golden-pinned — only the
//! standalone `CompoundPoisson` (see `golden_compound_poisson_streams`)
//! could be. `Merton`'s own jump driver is now seeded from the same
//! `Deterministic` the diffusion component uses, so it is pinnable too.
//! `golden_bates_streams` is the second: Task 2 of the same wave applied the
//! identical fix to `Bates1996`, whose jump term is *multiplicative*
//! (`sample_grid_relative_increments`, not `Merton`'s additive
//! `sample_grid_increments`) and whose output is a `[s, v]` pair rather than
//! a single array — both pinned below, with a same-file counterfactual
//! proving the pin is not diffusion-only.

use rand_distr::Normal;
use stochastic_rs::distributions::scalar::ScalarNormal;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::diffusion::fou::Fou;
use stochastic_rs::stochastic::diffusion::gbm::Gbm;
use stochastic_rs::stochastic::diffusion::ou::Ou;
use stochastic_rs::stochastic::jump::bates::Bates1996;
use stochastic_rs::stochastic::jump::merton::Merton;
use stochastic_rs::stochastic::noise::fgn::Fgn;
use stochastic_rs::stochastic::process::cpoisson::CompoundPoisson;
use stochastic_rs::stochastic::process::poisson::Poisson;
use stochastic_rs::stochastic::volatility::HestonPow;
use stochastic_rs::stochastic::volatility::heston::Heston;
use stochastic_rs::stochastic::volatility::sabr::Sabr;
use stochastic_rs::traits::PathSampler;
use stochastic_rs::traits::ProcessExt;

const N: usize = 8;

fn bits(a: &ndarray::Array1<f64>) -> Vec<u64> {
  a.iter().map(|x| x.to_bits()).collect()
}

/// Asserts each element of `actual` matches the golden value (stored as its
/// `f64` bit pattern) within a tolerance that absorbs cross-architecture
/// floating-point rounding while still catching any real behavioural drift —
/// a genuine change shifts a value far more than `1e-9`.
fn assert_close(actual: &ndarray::Array1<f64>, golden_bits: &[u64]) {
  assert_eq!(actual.len(), golden_bits.len(), "length mismatch");
  for (i, (&a, &gb)) in actual.iter().zip(golden_bits).enumerate() {
    let g = f64::from_bits(gb);
    let tol = 1e-9 * (1.0 + g.abs());
    assert!(
      (a - g).abs() <= tol,
      "index {i}: got {a}, golden {g}, |diff| {} > tol {tol}",
      (a - g).abs()
    );
  }
}

#[test]
fn golden_gbm_stream() {
  let gbm = Gbm::<f64, _>::new(0.05, 0.2, N, Some(1.0), Some(1.0), Deterministic::new(42));
  assert_close(
    &gbm.sample(),
    &[
      4607182418800017408,
      4607577785780026171,
      4607305368942984765,
      4606357005234000244,
      4606894812180602655,
      4606743184169166300,
      4607437034000694954,
      4607741774280140162,
    ],
  );
}

#[test]
fn golden_ou_stream() {
  let ou = Ou::<f64, _>::new(
    2.0,
    0.1,
    0.3,
    N,
    Some(0.5),
    Some(1.0),
    Deterministic::new(42),
  );
  assert_close(
    &ou.sample(),
    &[
      4602678819172646912,
      4602739020205830579,
      4599010430843072138,
      4588213546747600988,
      4594801757682296832,
      4592119808059091574,
      4598315888370368019,
      4599041240776096059,
    ],
  );
}

#[test]
fn golden_heston_streams() {
  let heston = Heston::<f64, _>::new(
    Some(1.0),
    Some(0.04),
    2.0,
    0.04,
    0.3,
    -0.7,
    0.05,
    N,
    Some(1.0),
    HestonPow::Sqrt,
    None,
    Deterministic::new(42),
  );
  let [s, v] = heston.sample();
  assert_close(
    &s,
    &[
      4607182418800017408,
      4607502780054079901,
      4607325004389578606,
      4607420592588706408,
      4606508958779655644,
      4606741005298078029,
      4607226524837684465,
      4606736510542841516,
    ],
  );
  assert_close(
    &v,
    &[
      4585925428558828667,
      4580662610146546399,
      4585546315862233050,
      4584983616031705446,
      4588547915715780940,
      4580640673470114080,
      4574353474520915898,
      4584221584215986456,
    ],
  );
}

#[test]
fn golden_sabr_streams() {
  // Pinned before the C1 correlation-fix rewrite lands (a reviewer verified
  // the rewrite's first path matches this pre-rewrite value for `Sabr` and
  // all 10 rewritten types), specifically so that rewrite cannot silently
  // shift it. `Sabr` is "clone-snapshot": `sampler()` currently does
  // `seed: self.seed.clone()`, so this exercises that shape directly.
  let sabr = Sabr::<f64, _>::new(
    0.3,
    0.5,
    -0.7,
    N,
    Some(1.0),
    Some(0.2),
    Some(1.0),
    Deterministic::new(42),
  );
  let [f, a] = sabr.sample();
  assert_close(
    &f,
    &[
      4607182418800017408,
      4607470611485312969,
      4607195393466663091,
      4607256618755914074,
      4606122964360972460,
      4606277885911825184,
      4606927526128110858,
      4605786245219509036,
    ],
  );
  assert_close(
    &a,
    &[
      4596373779694328218,
      4595561519371525225,
      4596207781262441424,
      4595996209766421720,
      4596855723141723989,
      4595755526112337061,
      4594915282930478293,
      4595904931460021670,
    ],
  );
}

#[test]
fn golden_fou_stream() {
  // Pinned before the C1 correlation-fix rewrite lands — see
  // `golden_sabr_streams` above. `Fou` is one of the 10 "lazy" types
  // rewritten to own a seed at `sampler()` construction.
  let fou = Fou::<f64, _>::new(
    0.7,
    1.3,
    0.8,
    0.2,
    N,
    Some(0.2),
    Some(1.0),
    Deterministic::new(42),
  );
  assert_close(
    &fou.sample(),
    &[
      4596373779694328218,
      4600306872643172956,
      4602099920409686161,
      4602782325076011741,
      4603273410930381009,
      4602687963748544572,
      4602116189115551230,
      4603097156562520143,
    ],
  );
}

/// `cum`/`jumps` (but not `times`) were re-pinned by the deterministic-
/// parallelism wave's cross-chunk-correlation fix: `CompoundPoisson`'s
/// `sampler()` now derives (not clones) its basis, and `Poisson::sample_impl`
/// consumes two internal ticks (`SimdExp::new` + `.rng()`) per call rather
/// than one. For a single-tick-per-call consumer, moving the derive from
/// per-path code to `sampler()` leaves the fed-in value unchanged (see
/// `golden_heston_streams`/`golden_sabr_streams`/`golden_fou_stream`); for a
/// two-tick consumer followed by more code that keeps reading the seed
/// (`cum`/`jumps` are drawn after `times`, from the same seed), the extra
/// tick that used to be absorbed inside a disposable cloned-then-derived
/// temporary is now visible on the shared counter, shifting everything
/// downstream. `times` is unaffected because nothing before it reads the
/// seed. This is expected and was traded for `sample_par`/`sample_map`
/// actually being cross-chunk-independent.
#[test]
fn golden_compound_poisson_streams() {
  let cpoisson = CompoundPoisson::<f64, _, _>::new(
    Normal::new(0.0, 0.1).unwrap(),
    Poisson::<f64, _>::new(0.5, Some(N), Some(1.0), Unseeded),
    Deterministic::new(44),
  );
  let [times, cum, jumps] = cpoisson.sample();
  assert_close(
    &times,
    &[
      0,
      4611384675406081356,
      4617286159363862215,
      4621901946075464258,
      4625674486717962014,
      4626419047739112712,
      4627097123718683255,
      4627422908561883916,
    ],
  );
  assert_close(
    &cum,
    &[
      0,
      13821590905287482682,
      13823140035420567097,
      13823193588762097129,
      13821607767976760320,
      13812879127933788120,
      13815136404819208294,
      13819433902103997129,
    ],
  );
  assert_close(
    &jumps,
    &[
      0,
      13821590905287482682,
      13814232978050278395,
      13792373287096932327,
      4591007703804512164,
      4595874907490768916,
      13808047411661410587,
      13814724200134044972,
    ],
  );
}

/// The first golden covering a jump chain — see this file's own header for
/// why `Merton` could not be included before the zero-exception-
/// reproducibility wave's Task 1. `lambda = 3.0` at `N = 8` (`dt = 1/7`)
/// gives `lambda * dt ≈ 0.43` per step, high enough that this stream
/// exercises at least one nonzero jump increment, not just an all-zero
/// `sample_grid_increments` short-circuit.
#[test]
fn golden_merton_streams() {
  let merton = Merton::new(
    0.03,
    0.2,
    3.0,
    0.0,
    ScalarNormal::new(0.0, 0.1),
    N,
    Some(0.0),
    Some(1.0),
    Deterministic::new(42),
  );
  assert_close(
    &merton.sample(),
    &[
      0,
      4590578534663088414,
      4581639020107344888,
      13815295515786926506,
      13809401478369850988,
      13804506849033497710,
      4590859968188903663,
      13785162989269995008,
    ],
  );
}

/// The second golden covering a jump chain (see this file's own header) and
/// the first covering `Bates1996`'s *multiplicative* jump term
/// (`sample_grid_relative_increments`, not `Merton`'s additive
/// `sample_grid_increments`) together with its `[s, v]` pair output.
/// `lambda = 3.0` at `N = 8` (`dt = 1/7`) matches `golden_merton_streams`'s
/// own reasoning: `lambda * dt ≈ 0.43` per step, high enough that this
/// stream exercises at least one nonzero jump increment. `k = 0.0` keeps the
/// drift's `-lambda*k` compensator term at zero regardless of `lambda`, so
/// the divergence proof below (comparing against a `lambda = 0`
/// counterfactual, same `k`) isolates the jump term specifically rather
/// than a coincidental drift shift.
#[test]
fn golden_bates_streams() {
  let bates = Bates1996::new(
    Some(0.05),
    None,
    None,
    None,
    3.0,
    0.0,
    0.04,
    1.5,
    0.3,
    -0.6,
    ScalarNormal::new(0.0, 0.1),
    N,
    Some(100.0),
    Some(0.04),
    Some(1.0),
    Some(false),
    Deterministic::new(42),
  );
  let [s, v] = bates.sample();
  assert_close(
    &s,
    &[
      4636737291354636288,
      4637237855814108933,
      4636980197197895274,
      4637119380743617754,
      4636347933066129122,
      4636396627911248212,
      4636650751471554252,
      4636854863732819059,
    ],
  );
  assert_close(
    &v,
    &[
      4585925428558828667,
      4579986716671955822,
      4584221907410356252,
      4582883765447896902,
      4586133690055868866,
      4572975989110218152,
      4561793493522807552,
      4577356478036536416,
    ],
  );

  // Not a diffusion-only pin: a lambda=0 counterfactual (same seed, same k,
  // same everything else) diverges on the price path, which is only
  // possible if the golden above actually depends on the jump term.
  let zero_lambda = Bates1996::new(
    Some(0.05),
    None,
    None,
    None,
    0.0,
    0.0,
    0.04,
    1.5,
    0.3,
    -0.6,
    ScalarNormal::new(0.0, 0.1),
    N,
    Some(100.0),
    Some(0.04),
    Some(1.0),
    Some(false),
    Deterministic::new(42),
  );
  let [s_zero, _] = zero_lambda.sample();
  assert_ne!(
    bits(&s),
    bits(&s_zero),
    "golden_bates_streams must not be reproducible solely via the diffusion \
     component — a lambda=0 counterfactual should diverge"
  );
}

#[test]
fn golden_fgn_stream() {
  let fgn = Fgn::new(0.7f64, N, None, Deterministic::new(42));
  assert_close(
    &fgn.sample(),
    &[
      4597197406585373975,
      13823117271691027942,
      13821044487902122082,
      13821397182329245168,
      4598861375682803509,
      4596990263574289093,
      13819492650888764188,
      4588986355184932690,
    ],
  );
}

#[test]
fn sampler_first_path_matches_sample() {
  // The first `sampler().sample()` reproduces `sample()` bit-for-bit because
  // both derive the Gaussian source identically from the same seed. This is a
  // same-machine comparison, so it stays exact and portable.
  let a = Gbm::<f64, _>::new(0.05, 0.2, 32, Some(1.0), Some(1.0), Deterministic::new(7));
  let b = Gbm::<f64, _>::new(0.05, 0.2, 32, Some(1.0), Some(1.0), Deterministic::new(7));
  assert_eq!(bits(&a.sample()), bits(&b.sampler().sample()));

  let f1 = Fgn::new(0.6f64, 32, Some(1.0), Deterministic::new(7));
  let f2 = Fgn::new(0.6f64, 32, Some(1.0), Deterministic::new(7));
  assert_eq!(bits(&f1.sample()), bits(&f2.sampler().sample()));
}

#[test]
fn sampler_continues_stream() {
  // Reusing a sampler yields a fresh, independent path on each call.
  let gbm = Gbm::<f64, _>::new(0.05, 0.2, 32, Some(1.0), Some(1.0), Deterministic::new(7));
  let mut s = gbm.sampler();
  let p1 = s.sample();
  let mut p2 = p1.clone();
  s.sample_into(&mut p2);
  assert_ne!(bits(&p1), bits(&p2));
}

#[test]
fn sample_map_matches_terminal_expectation() {
  // The buffer-reusing parallel fold must still produce the right law:
  // E[S_T] = exp(mu * T) = exp(0.05) for this GBM.
  let gbm = Gbm::<f64, _>::new(0.05, 0.2, 64, Some(1.0), Some(1.0), Unseeded);
  let m = 20_000;
  let terminals = gbm.sample_map(m, |p| *p.last().unwrap());
  assert_eq!(terminals.len(), m);
  let mean = terminals.iter().sum::<f64>() / m as f64;
  assert!((mean - 0.05f64.exp()).abs() < 0.01, "mean {mean}");
}

#[test]
fn sample_par_returns_distinct_paths() {
  // `sample_par` yields the requested count, each an independent realisation.
  let gbm = Gbm::<f64, _>::new(0.05, 0.2, 64, Some(1.0), Some(1.0), Unseeded);
  let paths = gbm.sample_par(64);
  assert_eq!(paths.len(), 64);
  assert!(paths.windows(2).all(|w| w[0] != w[1]));
}
