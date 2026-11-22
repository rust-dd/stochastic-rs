//! `Heston`'s `with_*` setters, generic over the scheme so they chain after `qe()` too; `HestonPow`
//! has no `PartialEq`, so it is compared with `matches!` outside the field snapshot.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::HestonPow;
use stochastic_rs_stochastic::volatility::heston::Euler;
use stochastic_rs_stochastic::volatility::heston::Heston;

fn heston_base_seeded<S: SeedExt>(seed: S) -> Heston<f64, Euler, S> {
  Heston::new(
    100.0,
    0.04,
    2.0,
    0.04,
    0.3,
    -0.7,
    0.05,
    64,
    Some(1.0),
    HestonPow::Sqrt,
    VarianceBoundary::Absorption,
    seed,
  )
}

fn heston_base() -> Heston<f64> {
  heston_base_seeded(Unseeded)
}

#[derive(Debug, PartialEq)]
struct HestonFields {
  s0: f64,
  v0: f64,
  kappa: f64,
  theta: f64,
  sigma: f64,
  rho: f64,
  mu: f64,
  n: usize,
  t: Option<f64>,
  boundary: VarianceBoundary,
}

fn fields<S: SeedExt>(x: &Heston<f64, Euler, S>) -> HestonFields {
  HestonFields {
    s0: x.s0,
    v0: x.v0,
    kappa: x.kappa,
    theta: x.theta,
    sigma: x.sigma,
    rho: x.rho,
    mu: x.mu,
    n: x.n,
    t: x.t,
    boundary: x.boundary,
  }
}

fn finite2(out: &[Array1<f64>; 2]) -> bool {
  out.iter().all(|a| a.iter().all(|v| v.is_finite()))
}

macro_rules! plain_test {
  ($name:ident, $setter:ident, $field:ident, $val:expr) => {
    #[test]
    fn $name() {
      let mut expected = heston_base();
      expected.$field = $val;
      let got = heston_base().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(fields(&got), fields(&expected));
      assert!(finite2(&got.sample()));
    }
  };
}

plain_test!(heston_with_s0_round_trip, with_s0, s0, 90.0);
plain_test!(heston_with_v0_round_trip, with_v0, v0, 0.06);
plain_test!(heston_with_mu_round_trip, with_mu, mu, 0.02);
plain_test!(
  heston_with_boundary_round_trip,
  with_boundary,
  boundary,
  VarianceBoundary::Reflect
);

#[test]
#[should_panic(expected = "kappa must satisfy `kappa >= T::zero()`, got kappa = -1.0")]
fn heston_with_kappa_rejects_negative() {
  let _ = heston_base().with_kappa(-1.0);
}

#[test]
#[should_panic(expected = "theta must satisfy `theta >= T::zero()`, got theta = -0.01")]
fn heston_with_theta_rejects_negative() {
  let _ = heston_base().with_theta(-0.01);
}

#[test]
#[should_panic(expected = "sigma must satisfy `sigma >= T::zero()`, got sigma = -0.1")]
fn heston_with_sigma_rejects_negative() {
  let _ = heston_base().with_sigma(-0.1);
}

#[test]
#[should_panic(expected = "v0 must satisfy `v0 >= T::zero()`, got v0 = -0.01")]
fn heston_with_v0_rejects_negative() {
  let _ = heston_base().with_v0(-0.01);
}

#[test]
fn heston_with_pow_round_trip() {
  let got = heston_base().with_pow(HestonPow::ThreeHalves);
  assert!(matches!(got.pow, HestonPow::ThreeHalves));
  assert_eq!(got.kappa, heston_base().kappa);
  assert!(finite2(&got.sample()));
}

#[test]
fn heston_with_rho_rebuilds_cgns_cache() {
  let mut expected = heston_base();
  expected.rho = -0.4;
  let got = heston_base().with_rho(-0.4);
  assert_eq!(got.rho, -0.4);
  assert_eq!(fields(&got), fields(&expected));

  let want = Heston::new(
    100.0,
    0.04,
    2.0,
    0.04,
    0.3,
    -0.4,
    0.05,
    64,
    Some(1.0),
    HestonPow::Sqrt,
    VarianceBoundary::Absorption,
    Deterministic::new(7),
  )
  .sample();
  let got_seeded = heston_base_seeded(Deterministic::new(7))
    .with_rho(-0.4)
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn heston_with_points_rebuilds_cgns_cache() {
  let mut expected = heston_base();
  expected.n = 128;
  let got = heston_base().with_points(128);
  assert_eq!(got.n, 128);
  assert_eq!(fields(&got), fields(&expected));
  assert!(finite2(&got.sample()));

  let want = Heston::new(
    100.0,
    0.04,
    2.0,
    0.04,
    0.3,
    -0.7,
    0.05,
    128,
    Some(1.0),
    HestonPow::Sqrt,
    VarianceBoundary::Absorption,
    Deterministic::new(9),
  )
  .sample();
  let got_seeded = heston_base_seeded(Deterministic::new(9))
    .with_points(128)
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn heston_with_horizon_rebuilds_cgns_cache() {
  let mut expected = heston_base();
  expected.t = Some(2.0);
  let got = heston_base().with_horizon(Some(2.0));
  assert_eq!(got.t, Some(2.0));
  assert_eq!(fields(&got), fields(&expected));

  let want = Heston::new(
    100.0,
    0.04,
    2.0,
    0.04,
    0.3,
    -0.7,
    0.05,
    64,
    Some(2.0),
    HestonPow::Sqrt,
    VarianceBoundary::Absorption,
    Deterministic::new(11),
  )
  .sample();
  let got_seeded = heston_base_seeded(Deterministic::new(11))
    .with_horizon(Some(2.0))
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn heston_with_seed_matches_fresh_construction() {
  let want = heston_base_seeded(Deterministic::new(13)).sample();
  let got = heston_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}

#[test]
fn heston_setters_chain_after_default_and_after_qe() {
  // `Heston::default().with_kappa(3.0).with_rho(-0.8)` type-checks and samples finitely on the
  // `Euler` default and after `.qe()`: the setters are generic over the scheme.
  let euler = Heston::<f64>::default().with_kappa(3.0).with_rho(-0.8);
  assert_eq!(euler.kappa, 3.0);
  assert_eq!(euler.rho, -0.8);
  assert!(finite2(&euler.sample()));

  let qe = Heston::<f64>::default().qe().with_kappa(3.0).with_rho(-0.8);
  assert_eq!(qe.kappa, 3.0);
  assert_eq!(qe.rho, -0.8);
  assert!(finite2(&qe.sample()));
}
