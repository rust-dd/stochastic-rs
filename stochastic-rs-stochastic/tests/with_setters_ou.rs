//! TDD tests for A1-c Task 4: `with_*` builder setters on `Ou`
//! (`diffusion/ou.rs`). No persisted cache: `sampler()` builds its Gaussian
//! stream fresh from `self.{theta,sigma,n,t,seed}` on every call.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::ou::Ou;
use stochastic_rs_stochastic::traits::ProcessExt;

#[derive(Debug, PartialEq)]
struct OuFields {
  theta: f64,
  mu: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
}

fn fields(x: &Ou<f64>) -> OuFields {
  OuFields {
    theta: x.theta,
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
      let expected = Ou::<f64> {
        $field: $val,
        ..Ou::<f64>::default()
      };
      let got = Ou::<f64>::default().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(fields(&got), fields(&expected));
      assert!(finite(&got.sample()));
    }
  };
}

plain_test!(ou_with_theta_round_trip, with_theta, theta, 3.0);
plain_test!(ou_with_mu_round_trip, with_mu, mu, 0.05);
plain_test!(ou_with_sigma_round_trip, with_sigma, sigma, 0.3);
plain_test!(ou_with_x0_round_trip, with_x0, x0, Some(1.0));
plain_test!(ou_with_steps_round_trip, with_steps, n, 64usize);
plain_test!(ou_with_horizon_round_trip, with_horizon, t, Some(2.0));

#[test]
fn ou_with_seed_matches_fresh_construction() {
  let want = Ou::new(
    2.0,
    0.0,
    0.2,
    64,
    Some(0.0),
    Some(1.0),
    Deterministic::new(13),
  )
  .sample();
  let got = Ou::new(
    2.0,
    0.0,
    0.2,
    64,
    Some(0.0),
    Some(1.0),
    Deterministic::new(1),
  )
  .with_seed(Deterministic::new(13))
  .sample();
  assert_eq!(want, got);
}
