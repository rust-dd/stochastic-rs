//! TDD tests for A1-c Task 4: `with_*` builder setters on
//! `DisplacedDiffusion` (`diffusion/displaced_diffusion.rs`). No persisted
//! cache: `sampler()` builds its Gaussian source fresh from
//! `self.{mu,sigma,n,t,seed}` on every call.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::displaced_diffusion::DisplacedDiffusion;
use stochastic_rs_stochastic::traits::ProcessExt;

#[derive(Debug, PartialEq)]
struct DdFields {
  mu: f64,
  sigma: f64,
  beta: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
}

fn fields(x: &DisplacedDiffusion<f64>) -> DdFields {
  DdFields {
    mu: x.mu,
    sigma: x.sigma,
    beta: x.beta,
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
      let expected = DisplacedDiffusion::<f64> {
        $field: $val,
        ..DisplacedDiffusion::<f64>::default()
      };
      let got = DisplacedDiffusion::<f64>::default().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(fields(&got), fields(&expected));
      assert!(finite(&got.sample()));
    }
  };
}

plain_test!(dd_with_mu_round_trip, with_mu, mu, 0.06);
plain_test!(dd_with_sigma_round_trip, with_sigma, sigma, 0.3);
plain_test!(dd_with_beta_round_trip, with_beta, beta, 10.0);
plain_test!(dd_with_x0_round_trip, with_x0, x0, Some(50.0));
plain_test!(dd_with_steps_round_trip, with_steps, n, 64usize);
plain_test!(dd_with_horizon_round_trip, with_horizon, t, Some(2.0));

#[test]
fn dd_with_seed_matches_fresh_construction() {
  let want = DisplacedDiffusion::new(
    0.05,
    0.2,
    30.0,
    64,
    Some(100.0),
    Some(1.0),
    Deterministic::new(13),
  )
  .sample();
  let got = DisplacedDiffusion::new(
    0.05,
    0.2,
    30.0,
    64,
    Some(100.0),
    Some(1.0),
    Deterministic::new(1),
  )
  .with_seed(Deterministic::new(13))
  .sample();
  assert_eq!(want, got);
}
