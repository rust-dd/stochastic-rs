//! TDD tests for A1-c Task 4: `with_*` builder setters on `Cev`
//! (`diffusion/cev.rs`). No persisted cache: `sampler()` builds its Gaussian
//! stream fresh from `self.{mu,sigma,gamma,n,t,seed}` on every call.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::cev::Cev;
use stochastic_rs_stochastic::traits::ProcessExt;

#[derive(Debug, PartialEq)]
struct CevFields {
  mu: f64,
  sigma: f64,
  gamma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
}

fn fields(x: &Cev<f64>) -> CevFields {
  CevFields {
    mu: x.mu,
    sigma: x.sigma,
    gamma: x.gamma,
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
      let expected = Cev::<f64> {
        $field: $val,
        ..Cev::<f64>::default()
      };
      let got = Cev::<f64>::default().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(fields(&got), fields(&expected));
      assert!(finite(&got.sample()));
    }
  };
}

plain_test!(cev_with_mu_round_trip, with_mu, mu, 0.06);
plain_test!(cev_with_sigma_round_trip, with_sigma, sigma, 0.3);
plain_test!(cev_with_gamma_round_trip, with_gamma, gamma, 0.5);
plain_test!(cev_with_x0_round_trip, with_x0, x0, Some(120.0));
plain_test!(cev_with_steps_round_trip, with_steps, n, 64usize);
plain_test!(cev_with_horizon_round_trip, with_horizon, t, Some(2.0));

#[test]
fn cev_with_seed_matches_fresh_construction() {
  let want = Cev::new(
    0.04,
    0.2,
    0.8,
    64,
    Some(1.0),
    Some(1.0),
    Deterministic::new(13),
  )
  .sample();
  let got = Cev::new(
    0.04,
    0.2,
    0.8,
    64,
    Some(1.0),
    Some(1.0),
    Deterministic::new(1),
  )
  .with_seed(Deterministic::new(13))
  .sample();
  assert_eq!(want, got);
}
