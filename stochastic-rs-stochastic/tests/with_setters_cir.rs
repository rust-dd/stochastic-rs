//! `with_*` setters on `Cir` (`diffusion/cir.rs`), which caches nothing: `sampler()` builds its
//! Gaussian stream from `self.{theta,sigma,n,t,seed}` on every call.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::traits::ProcessExt;

#[derive(Debug, PartialEq)]
struct CirFields {
  theta: f64,
  mu: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
  boundary: VarianceBoundary,
}

fn fields(x: &Cir<f64>) -> CirFields {
  CirFields {
    theta: x.theta,
    mu: x.mu,
    sigma: x.sigma,
    n: x.n,
    x0: x.x0,
    t: x.t,
    boundary: x.boundary,
  }
}

fn finite(out: &Array1<f64>) -> bool {
  out.iter().all(|v| v.is_finite())
}

macro_rules! plain_test {
  ($name:ident, $setter:ident, $field:ident, $val:expr) => {
    #[test]
    fn $name() {
      let expected = Cir::<f64> {
        $field: $val,
        ..Cir::<f64>::default()
      };
      let got = Cir::<f64>::default().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(fields(&got), fields(&expected));
      assert!(finite(&got.sample()));
    }
  };
}

plain_test!(cir_with_theta_round_trip, with_theta, theta, 3.0);
plain_test!(cir_with_mu_round_trip, with_mu, mu, 0.05);
plain_test!(cir_with_sigma_round_trip, with_sigma, sigma, 0.3);
plain_test!(cir_with_x0_round_trip, with_x0, x0, Some(0.05));
plain_test!(
  cir_with_boundary_round_trip,
  with_boundary,
  boundary,
  VarianceBoundary::Reflect
);
plain_test!(cir_with_points_round_trip, with_points, n, 64usize);
plain_test!(cir_with_horizon_round_trip, with_horizon, t, Some(2.0));

#[test]
fn cir_with_seed_matches_fresh_construction() {
  let want = Cir::new(
    2.5,
    0.04,
    0.2,
    64,
    Some(0.04),
    Some(1.0),
    VarianceBoundary::Absorption,
    Deterministic::new(13),
  )
  .sample();
  let got = Cir::new(
    2.5,
    0.04,
    0.2,
    64,
    Some(0.04),
    Some(1.0),
    VarianceBoundary::Absorption,
    Deterministic::new(1),
  )
  .with_seed(Deterministic::new(13))
  .sample();
  assert_eq!(want, got);
}
