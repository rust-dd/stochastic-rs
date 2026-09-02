//! TDD tests for A1-c Task 4: `with_*` builder setters on `BlackKarasinski`
//! (`interest/black_karasinski.rs`). No persisted cache: `sampler()`
//! computes its exact-OU decay/std fresh from `self.{a,t,n,seed}` on every
//! call. `theta: Fn1D<T>` is excluded from the generic field-equality
//! snapshot (no `PartialEq`) and checked by calling it directly instead.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::interest::black_karasinski::BlackKarasinski;
use stochastic_rs_stochastic::traits::ProcessExt;

fn theta0(_t: f64) -> f64 {
  0.05
}

fn theta1(_t: f64) -> f64 {
  0.08
}

#[derive(Debug, PartialEq)]
struct BkFields {
  a: f64,
  sigma: f64,
  n: usize,
  r0: Option<f64>,
  t: Option<f64>,
}

fn fields(x: &BlackKarasinski<f64>) -> BkFields {
  BkFields {
    a: x.a,
    sigma: x.sigma,
    n: x.n,
    r0: x.r0,
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
      let expected = BlackKarasinski::<f64> {
        $field: $val,
        ..BlackKarasinski::<f64>::default()
      };
      let got = BlackKarasinski::<f64>::default().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(fields(&got), fields(&expected));
      assert!(finite(&got.sample()));
    }
  };
}

plain_test!(bk_with_a_round_trip, with_a, a, 0.6);
plain_test!(bk_with_sigma_round_trip, with_sigma, sigma, 0.15);
plain_test!(bk_with_r0_round_trip, with_r0, r0, Some(0.04));
plain_test!(bk_with_steps_round_trip, with_steps, n, 64usize);
plain_test!(bk_with_horizon_round_trip, with_horizon, t, Some(2.0));

#[test]
fn bk_with_theta_round_trip() {
  let got = BlackKarasinski::<f64>::default().with_theta(theta1 as fn(f64) -> f64);
  assert_eq!(got.theta.call(0.0), 0.08);
  assert_eq!(got.a, BlackKarasinski::<f64>::default().a);
  assert_eq!(got.sigma, BlackKarasinski::<f64>::default().sigma);
  assert!(finite(&got.sample()));
}

#[test]
fn bk_with_seed_matches_fresh_construction() {
  let want = BlackKarasinski::new(
    theta0 as fn(f64) -> f64,
    0.8,
    0.1,
    64,
    Some(0.03),
    Some(1.0),
    Deterministic::new(13),
  )
  .sample();
  let got = BlackKarasinski::new(
    theta0 as fn(f64) -> f64,
    0.8,
    0.1,
    64,
    Some(0.03),
    Some(1.0),
    Deterministic::new(1),
  )
  .with_seed(Deterministic::new(13))
  .sample();
  assert_eq!(want, got);
}
