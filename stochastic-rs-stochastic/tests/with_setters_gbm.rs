//! `Gbm`'s setters: those feeding the terminal log-normal cache (`mu`, `sigma`, `x0`, `t`)
//! recompute it, read back through `DistributionExt::mean`; `with_points` and `with_seed` leave it.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::traits::DistributionExt;
use stochastic_rs_stochastic::traits::ProcessExt;

fn gbm_base_seeded<S: SeedExt>(seed: S) -> Gbm<f64, S> {
  Gbm::new(0.05, 0.2, 64, 100.0, Some(1.0), seed)
}

fn gbm_base() -> Gbm<f64> {
  gbm_base_seeded(Unseeded)
}

#[derive(Debug, PartialEq)]
struct GbmFields {
  mu: f64,
  sigma: f64,
  n: usize,
  x0: f64,
  t: Option<f64>,
}

fn fields<S: SeedExt>(x: &Gbm<f64, S>) -> GbmFields {
  GbmFields {
    mu: x.mu,
    sigma: x.sigma,
    n: x.n,
    x0: x.x0,
    t: x.t,
  }
}

fn finite(out: &Array1<f64>) -> bool {
  out.iter().all(|v| v.is_finite())
}

macro_rules! plain_test {
  ($name:ident, $setter:ident, $field:ident, $val:expr) => {
    #[test]
    fn $name() {
      let mut expected = gbm_base();
      expected.$field = $val;
      let got = gbm_base().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(fields(&got), fields(&expected));
      assert!(finite(&got.sample()));
    }
  };
}

plain_test!(gbm_with_points_round_trip, with_points, n, 128usize);

#[test]
fn gbm_with_mu_rebuilds_terminal_lognormal_cache() {
  let mut expected = gbm_base();
  expected.mu = 0.08;
  let got = gbm_base().with_mu(0.08);
  assert_eq!(got.mu, 0.08);
  assert_eq!(fields(&got), fields(&expected));

  let want = Gbm::new(0.08, 0.2, 64, 100.0, Some(1.0), Unseeded);
  // Cache-value check: `mean()` reads `ln_mu`/`ln_sigma` directly, so a
  // stale cache (still keyed on the old mu=0.05) would disagree here even
  // though `got.mu` itself already reads correctly.
  assert_eq!(got.mean(), want.mean());

  let want_seeded = Gbm::new(0.08, 0.2, 64, 100.0, Some(1.0), Deterministic::new(7)).sample();
  let got_seeded = gbm_base_seeded(Deterministic::new(7))
    .with_mu(0.08)
    .sample();
  assert_eq!(want_seeded, got_seeded);
}

#[test]
fn gbm_with_sigma_rebuilds_terminal_lognormal_cache() {
  let mut expected = gbm_base();
  expected.sigma = 0.35;
  let got = gbm_base().with_sigma(0.35);
  assert_eq!(got.sigma, 0.35);
  assert_eq!(fields(&got), fields(&expected));

  let want = Gbm::new(0.05, 0.35, 64, 100.0, Some(1.0), Unseeded);
  assert_eq!(got.mean(), want.mean());
  assert_eq!(got.variance(), want.variance());
}

#[test]
fn gbm_with_x0_rebuilds_terminal_lognormal_cache() {
  let mut expected = gbm_base();
  expected.x0 = 50.0;
  let got = gbm_base().with_x0(50.0);
  assert_eq!(got.x0, 50.0);
  assert_eq!(fields(&got), fields(&expected));

  let want = Gbm::new(0.05, 0.2, 64, 50.0, Some(1.0), Unseeded);
  assert_eq!(got.mean(), want.mean());
}

#[test]
fn gbm_with_horizon_rebuilds_terminal_lognormal_cache() {
  let mut expected = gbm_base();
  expected.t = Some(2.0);
  let got = gbm_base().with_horizon(Some(2.0));
  assert_eq!(got.t, Some(2.0));
  assert_eq!(fields(&got), fields(&expected));

  let want = Gbm::new(0.05, 0.2, 64, 100.0, Some(2.0), Unseeded);
  assert_eq!(got.mean(), want.mean());
  assert_eq!(got.variance(), want.variance());
}

#[test]
fn gbm_with_seed_matches_fresh_construction() {
  let want = gbm_base_seeded(Deterministic::new(13)).sample();
  let got = gbm_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}

/// `T::default().with_x(v)` round-trip rooted in `Default`, not the
/// `gbm_base()` helper the bit-exact cache tests above need (those require
/// literal, hand-chosen fixtures for the fresh-construction comparisons;
/// this one exercises the wave's own headline "same model, one parameter
/// changed" form directly). Compares via the `GbmFields` mirror (not a
/// `Gbm { .., ..Gbm::default() }` struct-update literal): `Gbm` has private
/// cache fields (`ln_mu`/`ln_sigma`), so that literal wouldn't even compile
/// from this external test crate.
#[test]
fn gbm_default_with_x0_round_trip() {
  let base = Gbm::<f64>::default();
  let got = Gbm::<f64>::default().with_x0(50.0);
  let expected = GbmFields {
    x0: 50.0,
    ..fields(&base)
  };
  assert_eq!(got.x0, 50.0);
  assert_eq!(fields(&got), expected);
  assert!(finite(&got.sample()));
}
