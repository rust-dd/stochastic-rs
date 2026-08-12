//! TDD tests for A1-c Task 4: `with_*` builder setters on `Heston`
//! (`volatility/heston.rs`). Cache: private `cgns: Cgns<T>` keyed on
//! `(rho, n, t)`, same shape as the previous wave's `BatesSvj`/`Hkde`/etc.
//! Setters are implemented generically over `Sch: HestonScheme` (not just
//! `Euler`, the only scheme `new()`/`Default` produce) since `cgns` and
//! every other field are scheme-independent — this is what lets
//! `Heston::default().with_kappa(3.0).with_rho(-0.8)` (this wave's own
//! headline example) type-check at all, since `default()` returns
//! `Heston<T, Unseeded, Euler>` and the setters must accept that receiver.
//! `HestonPow` has no `PartialEq`, so it is excluded from the generic
//! field-equality snapshot and compared via `matches!` directly instead.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::HestonPow;
use stochastic_rs_stochastic::volatility::heston::Heston;

fn heston_base_seeded<S: SeedExt>(seed: S) -> Heston<f64, S> {
  Heston::new(
    Some(100.0),
    Some(0.04),
    2.0,
    0.04,
    0.3,
    -0.7,
    0.05,
    64,
    Some(1.0),
    HestonPow::Sqrt,
    Some(false),
    seed,
  )
}
fn heston_base() -> Heston<f64> {
  heston_base_seeded(Unseeded)
}

#[derive(Debug, PartialEq)]
struct HestonFields {
  s0: Option<f64>,
  v0: Option<f64>,
  kappa: f64,
  theta: f64,
  sigma: f64,
  rho: f64,
  mu: f64,
  n: usize,
  t: Option<f64>,
  use_sym: Option<bool>,
}
fn fields<S: SeedExt>(x: &Heston<f64, S>) -> HestonFields {
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
    use_sym: x.use_sym,
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

plain_test!(heston_with_s0_round_trip, with_s0, s0, Some(90.0));
plain_test!(heston_with_v0_round_trip, with_v0, v0, Some(0.06));
plain_test!(heston_with_mu_round_trip, with_mu, mu, 0.02);
plain_test!(
  heston_with_use_sym_round_trip,
  with_use_sym,
  use_sym,
  Some(true)
);

#[test]
#[should_panic(expected = "kappa must be non-negative")]
fn heston_with_kappa_rejects_negative() {
  let _ = heston_base().with_kappa(-1.0);
}

#[test]
#[should_panic(expected = "theta must be non-negative")]
fn heston_with_theta_rejects_negative() {
  let _ = heston_base().with_theta(-0.01);
}

#[test]
#[should_panic(expected = "sigma must be non-negative")]
fn heston_with_sigma_rejects_negative() {
  let _ = heston_base().with_sigma(-0.1);
}

#[test]
#[should_panic(expected = "v0 must be non-negative")]
fn heston_with_v0_rejects_negative() {
  let _ = heston_base().with_v0(Some(-0.01));
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
    Some(100.0),
    Some(0.04),
    2.0,
    0.04,
    0.3,
    -0.4,
    0.05,
    64,
    Some(1.0),
    HestonPow::Sqrt,
    Some(false),
    Deterministic::new(7),
  )
  .sample();
  let got_seeded = heston_base_seeded(Deterministic::new(7))
    .with_rho(-0.4)
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn heston_with_steps_rebuilds_cgns_cache() {
  let mut expected = heston_base();
  expected.n = 128;
  let got = heston_base().with_steps(128);
  assert_eq!(got.n, 128);
  assert_eq!(fields(&got), fields(&expected));
  assert!(finite2(&got.sample()));

  let want = Heston::new(
    Some(100.0),
    Some(0.04),
    2.0,
    0.04,
    0.3,
    -0.7,
    0.05,
    128,
    Some(1.0),
    HestonPow::Sqrt,
    Some(false),
    Deterministic::new(9),
  )
  .sample();
  let got_seeded = heston_base_seeded(Deterministic::new(9))
    .with_steps(128)
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
    Some(100.0),
    Some(0.04),
    2.0,
    0.04,
    0.3,
    -0.7,
    0.05,
    64,
    Some(2.0),
    HestonPow::Sqrt,
    Some(false),
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
  // The wave's own headline example: `Heston::default().with_kappa(3.0)
  // .with_rho(-0.8)` must type-check and sample finitely, both for the
  // `Euler`-scheme `default()` directly and after switching schemes via
  // `.qe()` (setters are scheme-generic, not `Euler`-only).
  let euler = Heston::<f64>::default().with_kappa(3.0).with_rho(-0.8);
  assert_eq!(euler.kappa, 3.0);
  assert_eq!(euler.rho, -0.8);
  assert!(finite2(&euler.sample()));

  let qe = Heston::<f64>::default().qe().with_kappa(3.0).with_rho(-0.8);
  assert_eq!(qe.kappa, 3.0);
  assert_eq!(qe.rho, -0.8);
  assert!(finite2(&qe.sample()));
}
