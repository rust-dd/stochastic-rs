//! TDD tests for A1-c Task 4: `with_*` builder setters on `HullWhite`
//! (`interest/hull_white.rs`). No persisted cache: `sampler()` builds its
//! Gaussian source fresh from `self.{n,t,seed}` and borrows `&self.theta`
//! directly on every call. `theta: Fn1D<T>` has no `PartialEq` (nor
//! `Fn1D` at all), so it is excluded from the generic field-equality
//! snapshot and instead checked by calling it in `with_theta`'s own test.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::interest::hull_white::HullWhite;
use stochastic_rs_stochastic::traits::ProcessExt;

fn theta0(_t: f64) -> f64 {
  0.04
}
fn theta1(_t: f64) -> f64 {
  0.07
}

#[derive(Debug, PartialEq)]
struct HwFields {
  alpha: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
}
fn fields(x: &HullWhite<f64>) -> HwFields {
  HwFields {
    alpha: x.alpha,
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
      let expected = HullWhite::<f64> {
        $field: $val,
        ..HullWhite::<f64>::default()
      };
      let got = HullWhite::<f64>::default().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(fields(&got), fields(&expected));
      assert!(finite(&got.sample()));
    }
  };
}

plain_test!(hw_with_alpha_round_trip, with_alpha, alpha, 0.6);
plain_test!(hw_with_sigma_round_trip, with_sigma, sigma, 0.04);
plain_test!(hw_with_x0_round_trip, with_x0, x0, Some(0.03));
plain_test!(hw_with_steps_round_trip, with_steps, n, 64usize);
plain_test!(hw_with_horizon_round_trip, with_horizon, t, Some(2.0));

#[test]
fn hw_with_theta_round_trip() {
  let got = HullWhite::<f64>::default().with_theta(theta1 as fn(f64) -> f64);
  assert_eq!(got.theta.call(0.0), 0.07);
  assert_eq!(got.alpha, HullWhite::<f64>::default().alpha);
  assert_eq!(got.sigma, HullWhite::<f64>::default().sigma);
  assert!(finite(&got.sample()));
}

#[test]
fn hw_with_seed_matches_fresh_construction() {
  let want = HullWhite::new(
    theta0 as fn(f64) -> f64,
    0.4,
    0.02,
    64,
    Some(0.02),
    Some(1.0),
    Deterministic::new(13),
  )
  .sample();
  let got = HullWhite::new(
    theta0 as fn(f64) -> f64,
    0.4,
    0.02,
    64,
    Some(0.02),
    Some(1.0),
    Deterministic::new(1),
  )
  .with_seed(Deterministic::new(13))
  .sample();
  assert_eq!(want, got);
}
