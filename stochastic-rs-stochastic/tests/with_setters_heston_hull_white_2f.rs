//! `HestonHullWhite2F`'s `with_*` setters, generic over the scheme so they chain after `qe()` too.

use ndarray::Array2;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::heston::Euler;
use stochastic_rs_stochastic::volatility::heston_hull_white_2f::HestonHullWhite2F;

fn theta_r(_t: f64) -> f64 {
  0.033
}

fn rho() -> Array2<f64> {
  ndarray::array![
    [1.0, -0.3, 0.35, 0.08],
    [-0.3, 1.0, 0.0, 0.0],
    [0.35, 0.0, 1.0, -0.4],
    [0.08, 0.0, -0.4, 1.0],
  ]
}

fn base() -> HestonHullWhite2F<f64, Euler, Deterministic> {
  HestonHullWhite2F::new(
    1.0,
    0.2,
    0.8,
    0.2,
    0.2,
    theta_r as fn(f64) -> f64,
    1.1,
    0.01,
    0.015,
    0.8,
    rho(),
    Some(0.03),
    32,
    Some(1.0),
    Deterministic::new(5),
  )
}

macro_rules! round_trip {
  ($name:ident, $setter:ident, $field:ident, $val:expr) => {
    #[test]
    fn $name() {
      let got = base().$setter($val);
      assert_eq!(got.$field, $val);
      assert_ne!(
        got.sample(),
        base().sample(),
        "the sampler reads the new value"
      );
      let qe = base().qe().$setter($val);
      assert_eq!(qe.$field, $val);
      assert_ne!(qe.sample(), base().qe().sample());
    }
  };
}

round_trip!(with_s0_round_trip, with_s0, s0, 90.0);
round_trip!(with_v0_round_trip, with_v0, v0, 0.06);
round_trip!(with_kappa_round_trip, with_kappa, kappa, 1.4);
round_trip!(with_theta_round_trip, with_theta, theta, 0.1);
round_trip!(with_sigma_round_trip, with_sigma, sigma, 0.35);
round_trip!(with_a_round_trip, with_a, a, 0.4);
round_trip!(with_sigma1_round_trip, with_sigma1, sigma1, 0.03);
round_trip!(with_sigma2_round_trip, with_sigma2, sigma2, 0.03);
round_trip!(with_r0_round_trip, with_r0, r0, Some(0.01));
round_trip!(with_points_round_trip, with_points, n, 48);
round_trip!(with_horizon_round_trip, with_horizon, t, Some(2.0));

#[test]
fn with_b_reaches_the_auxiliary_factor() {
  let got = base().with_b(0.2);
  assert_eq!(got.b, 0.2);
  assert_ne!(got.sample().u, base().sample().u);
}

#[test]
fn with_theta_r_reaches_the_rate() {
  fn steeper(t: f64) -> f64 {
    0.033 + 0.02 * t
  }
  let got = base().with_theta_r(steeper as fn(f64) -> f64).sample();
  assert_ne!(got.r, base().sample().r);
}

#[test]
fn with_seed_replaces_the_stream() {
  let got = base().with_seed(Deterministic::new(6));
  assert_ne!(got.sample(), base().sample());
}

#[test]
#[should_panic(expected = "v0 must satisfy `v0 >= T::zero()`, got v0 = -0.01")]
fn with_v0_rejects_negative() {
  let _ = base().with_v0(-0.01);
}

#[test]
#[should_panic(expected = "sigma2 must satisfy `sigma2 >= T::zero()`, got sigma2 = -0.1")]
fn with_sigma2_rejects_negative() {
  let _ = base().with_sigma2(-0.1);
}

#[test]
#[should_panic(expected = "a must satisfy `a.is_finite()`, got a = NaN")]
fn with_a_rejects_nan() {
  let _ = base().with_a(f64::NAN);
}
