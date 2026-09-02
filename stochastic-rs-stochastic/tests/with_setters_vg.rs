//! TDD tests for A1-c Task 4: `with_*` builder setters on `Vg`
//! (`jump/vg.rs`). No persisted cache: `sampler()` builds its gamma
//! subordinator and Gaussian source fresh from `self.{nu,n,t,seed}` on
//! every call. `new()`'s one assert (`nu > 0`) is replicated in `with_nu`.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::jump::vg::Vg;
use stochastic_rs_stochastic::traits::ProcessExt;

#[derive(Debug, PartialEq)]
struct VgFields {
  mu: f64,
  sigma: f64,
  nu: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
}

fn fields(x: &Vg<f64>) -> VgFields {
  VgFields {
    mu: x.mu,
    sigma: x.sigma,
    nu: x.nu,
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
      let expected = Vg::<f64> {
        $field: $val,
        ..Vg::<f64>::default()
      };
      let got = Vg::<f64>::default().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(fields(&got), fields(&expected));
      assert!(finite(&got.sample()));
    }
  };
}

plain_test!(vg_with_mu_round_trip, with_mu, mu, 0.05);
plain_test!(vg_with_sigma_round_trip, with_sigma, sigma, 0.3);
plain_test!(vg_with_nu_round_trip, with_nu, nu, 0.25);
plain_test!(vg_with_x0_round_trip, with_x0, x0, Some(1.0));
plain_test!(vg_with_steps_round_trip, with_steps, n, 64usize);
plain_test!(vg_with_horizon_round_trip, with_horizon, t, Some(2.0));

#[test]
#[should_panic(expected = "nu must be positive")]
fn vg_with_nu_rejects_non_positive() {
  let _ = Vg::<f64>::default().with_nu(0.0);
}

#[test]
fn vg_with_seed_matches_fresh_construction() {
  let want = Vg::new(
    0.0,
    0.2,
    0.15,
    64,
    Some(0.0),
    Some(1.0),
    Deterministic::new(13),
  )
  .sample();
  let got = Vg::new(
    0.0,
    0.2,
    0.15,
    64,
    Some(0.0),
    Some(1.0),
    Deterministic::new(1),
  )
  .with_seed(Deterministic::new(13))
  .sample();
  assert_eq!(want, got);
}
