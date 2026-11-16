//! The `with_*` setters of `FbatesSvj`, which keeps no cache: `sampler()` builds its `cgns`.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::fbates_svj::FbatesSvj;

fn finite2(out: &[Array1<f64>; 2]) -> bool {
  out.iter().all(|a| a.iter().all(|v| v.is_finite()))
}

fn fbsvj_base_seeded<S: SeedExt>(seed: S) -> FbatesSvj<f64, S> {
  FbatesSvj::new(
    0.1,
    0.05,
    100.0,
    0.04,
    0.04,
    2.0,
    0.3,
    -0.7,
    0.5,
    -0.01,
    0.1,
    256,
    Some(1.0),
    seed,
  )
}

fn fbsvj_base() -> FbatesSvj<f64> {
  fbsvj_base_seeded(Unseeded)
}

#[derive(Debug, PartialEq)]
struct FbsvjFields {
  hurst: f64,
  mu: f64,
  s0: f64,
  v0: f64,
  theta: f64,
  kappa: f64,
  xi: f64,
  rho: f64,
  lambda: f64,
  nu: f64,
  omega: f64,
  n: usize,
  t: Option<f64>,
}

fn fbsvj_fields<S: SeedExt>(x: &FbatesSvj<f64, S>) -> FbsvjFields {
  FbsvjFields {
    hurst: x.hurst,
    mu: x.mu,
    s0: x.s0,
    v0: x.v0,
    theta: x.theta,
    kappa: x.kappa,
    xi: x.xi,
    rho: x.rho,
    lambda: x.lambda,
    nu: x.nu,
    omega: x.omega,
    n: x.n,
    t: x.t,
  }
}

macro_rules! fbsvj_plain_test {
  ($name:ident, $setter:ident, $field:ident, $val:expr) => {
    #[test]
    fn $name() {
      let mut expected = fbsvj_base();
      expected.$field = $val;
      let got = fbsvj_base().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(fbsvj_fields(&got), fbsvj_fields(&expected));
      assert!(finite2(&got.sample()));
    }
  };
}

fbsvj_plain_test!(fbates_svj_with_hurst_round_trip, with_hurst, hurst, 0.15);
fbsvj_plain_test!(fbates_svj_with_mu_round_trip, with_mu, mu, 0.08);
fbsvj_plain_test!(fbates_svj_with_s0_round_trip, with_s0, s0, 120.0);
fbsvj_plain_test!(fbates_svj_with_v0_round_trip, with_v0, v0, 0.05);
fbsvj_plain_test!(fbates_svj_with_theta_round_trip, with_theta, theta, 0.05);
fbsvj_plain_test!(fbates_svj_with_kappa_round_trip, with_kappa, kappa, 2.5);
fbsvj_plain_test!(fbates_svj_with_xi_round_trip, with_xi, xi, 0.4);
fbsvj_plain_test!(fbates_svj_with_rho_round_trip, with_rho, rho, -0.4);
fbsvj_plain_test!(fbates_svj_with_lambda_round_trip, with_lambda, lambda, 0.8);
fbsvj_plain_test!(fbates_svj_with_nu_round_trip, with_nu, nu, -0.02);
fbsvj_plain_test!(fbates_svj_with_omega_round_trip, with_omega, omega, 0.15);
fbsvj_plain_test!(fbates_svj_with_points_round_trip, with_points, n, 64usize);
fbsvj_plain_test!(
  fbates_svj_with_horizon_round_trip,
  with_horizon,
  t,
  Some(2.0)
);

#[test]
fn fbates_svj_with_seed_matches_fresh_construction() {
  let want = fbsvj_base_seeded(Deterministic::new(13)).sample();
  let got = fbsvj_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}
