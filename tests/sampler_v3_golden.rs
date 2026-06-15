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
//! `Merton` is intentionally absent: it hard-wires its inner
//! `CompoundPoisson<T, D>` to `Unseeded`, so its jump chain is not
//! bit-reproducible; the standalone `CompoundPoisson` covers the jump family.

use rand_distr::Normal;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::diffusion::gbm::Gbm;
use stochastic_rs::stochastic::diffusion::ou::Ou;
use stochastic_rs::stochastic::noise::fgn::Fgn;
use stochastic_rs::stochastic::process::cpoisson::CompoundPoisson;
use stochastic_rs::stochastic::process::poisson::Poisson;
use stochastic_rs::stochastic::volatility::HestonPow;
use stochastic_rs::stochastic::volatility::heston::Heston;
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
      13813246270254564078,
      13813493359277356966,
      13816202492603102966,
      13809209142784623622,
      13815935968900354872,
      13818888094263487897,
      13821817264823823442,
    ],
  );
  assert_close(
    &jumps,
    &[
      0,
      13813246270254564078,
      13793425308110460687,
      13808951124542062240,
      4589992313301300467,
      13813097826453328181,
      13812833020371879930,
      13814434815486154264,
    ],
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
