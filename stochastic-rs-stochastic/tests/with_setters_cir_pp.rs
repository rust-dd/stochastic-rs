//! TDD tests for A1-c Task 4: `with_*` builder setters on `CirPlusPlus`
//! (`interest/cir_pp.rs`). No persisted cache: `sampler()` builds a
//! transient `Cir` (and its sampler) fresh from `self`'s own fields on
//! every call — there is no stored `Cir`/`CirSampler` field on the struct
//! itself. `phi: Fn1D<T>` is excluded from the generic field-equality
//! snapshot (no `PartialEq`) and checked by calling it directly instead.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::interest::cir_pp::CirPlusPlus;
use stochastic_rs_stochastic::traits::ProcessExt;

fn phi0(_t: f64) -> f64 {
  0.0
}
fn phi1(_t: f64) -> f64 {
  0.03
}

#[derive(Debug, PartialEq)]
struct PpFields {
  kappa: f64,
  theta: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
  use_sym: Option<bool>,
}
fn fields(x: &CirPlusPlus<f64>) -> PpFields {
  PpFields {
    kappa: x.kappa,
    theta: x.theta,
    sigma: x.sigma,
    n: x.n,
    x0: x.x0,
    t: x.t,
    use_sym: x.use_sym,
  }
}
fn finite(out: &Array1<f64>) -> bool {
  out.iter().all(|v| v.is_finite())
}

macro_rules! plain_test {
  ($name:ident, $setter:ident, $field:ident, $val:expr) => {
    #[test]
    fn $name() {
      let expected = CirPlusPlus::<f64> {
        $field: $val,
        ..CirPlusPlus::<f64>::default()
      };
      let got = CirPlusPlus::<f64>::default().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(fields(&got), fields(&expected));
      assert!(finite(&got.sample()));
    }
  };
}

plain_test!(pp_with_kappa_round_trip, with_kappa, kappa, 3.0);
plain_test!(pp_with_theta_round_trip, with_theta, theta, 0.05);
plain_test!(pp_with_sigma_round_trip, with_sigma, sigma, 0.3);
plain_test!(pp_with_x0_round_trip, with_x0, x0, Some(0.05));
plain_test!(
  pp_with_use_sym_round_trip,
  with_use_sym,
  use_sym,
  Some(true)
);
plain_test!(pp_with_steps_round_trip, with_steps, n, 64usize);
plain_test!(pp_with_horizon_round_trip, with_horizon, t, Some(2.0));

#[test]
fn pp_with_phi_round_trip() {
  let got = CirPlusPlus::<f64>::default().with_phi(phi1 as fn(f64) -> f64);
  assert_eq!(got.phi.call(0.0), 0.03);
  assert_eq!(got.kappa, CirPlusPlus::<f64>::default().kappa);
  assert!(finite(&got.sample()));
}

#[test]
fn pp_with_seed_matches_fresh_construction() {
  let want = CirPlusPlus::new(
    2.5,
    0.04,
    0.2,
    phi0 as fn(f64) -> f64,
    64,
    Some(0.04),
    Some(1.0),
    None,
    Deterministic::new(13),
  )
  .sample();
  let got = CirPlusPlus::new(
    2.5,
    0.04,
    0.2,
    phi0 as fn(f64) -> f64,
    64,
    Some(0.04),
    Some(1.0),
    None,
    Deterministic::new(1),
  )
  .with_seed(Deterministic::new(13))
  .sample();
  assert_eq!(want, got);
}
