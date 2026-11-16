//! `Merton`'s setters round-trip and match a fresh `new(..)` bit for bit, jumps included.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::jump::merton::Merton;
use stochastic_rs_stochastic::traits::ProcessExt;

fn merton_base_seeded<S: SeedExt>(seed: S) -> Merton<f64, SimdNormal<f64>, S> {
  Merton::new(
    0.03,
    0.2,
    1.0,
    0.0,
    SimdNormal::new(0.0, 0.1),
    64,
    Some(0.0),
    Some(1.0),
    seed,
  )
}

fn merton_base() -> Merton<f64, SimdNormal<f64>> {
  merton_base_seeded(Unseeded)
}

#[derive(Debug, PartialEq)]
struct MertonFields {
  alpha: f64,
  sigma: f64,
  lambda: f64,
  theta: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
}

fn fields<S: SeedExt>(x: &Merton<f64, SimdNormal<f64>, S>) -> MertonFields {
  MertonFields {
    alpha: x.alpha,
    sigma: x.sigma,
    lambda: x.lambda,
    theta: x.theta,
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
      let mut expected = merton_base();
      expected.$field = $val;
      let got = merton_base().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(fields(&got), fields(&expected));
      assert!(finite(&got.sample()));
    }
  };
}

plain_test!(merton_with_alpha_round_trip, with_alpha, alpha, 0.06);
plain_test!(merton_with_sigma_round_trip, with_sigma, sigma, 0.3);
plain_test!(merton_with_lambda_round_trip, with_lambda, lambda, 2.0);
plain_test!(merton_with_theta_round_trip, with_theta, theta, 0.02);
plain_test!(merton_with_x0_round_trip, with_x0, x0, Some(1.0));

#[test]
fn merton_with_jump_law_round_trip() {
  let got = merton_base().with_jump_law(SimdNormal::new(0.0, 5.0));
  assert_eq!(fields(&got), fields(&merton_base()));
  assert_eq!(got.jump_law().sigma(), 5.0);
  assert!(finite(&got.sample()));
}

fn merton_at(lambda: f64, law: SimdNormal<f64>) -> Merton<f64, SimdNormal<f64>, Deterministic> {
  Merton::new(
    0.03,
    0.2,
    lambda,
    0.0,
    law,
    64,
    Some(0.0),
    Some(1.0),
    Deterministic::new(3),
  )
}

/// A zero rate draws no jumps at all, so the comparison with it is exact whatever the seed.
#[test]
fn merton_with_lambda_matches_fresh_construction() {
  let law = SimdNormal::new(0.0, 0.1);
  let got = merton_at(0.0, law).with_lambda(80.0).sample();
  assert_eq!(got, merton_at(80.0, law).sample());
  assert_ne!(got, merton_at(0.0, law).sample());
  assert_eq!(
    merton_at(80.0, law).with_lambda(0.0).sample(),
    merton_at(0.0, law).sample()
  );
}

#[test]
fn merton_with_jump_law_matches_fresh_construction() {
  let wide = SimdNormal::new(0.0, 0.5);
  let got = merton_at(80.0, SimdNormal::new(0.0, 0.1)).with_jump_law(wide);
  assert_eq!(got.sample(), merton_at(80.0, wide).sample());
}

#[test]
fn merton_with_points_matches_fresh_construction() {
  let mut expected = merton_base_seeded(Unseeded);
  expected.n = 128;
  let got = merton_base_seeded(Unseeded).with_points(128);
  assert_eq!(got.n, 128);
  assert_eq!(fields(&got), fields(&expected));

  let want = Merton::new(
    0.03,
    0.2,
    1.0,
    0.0,
    SimdNormal::new(0.0, 0.1),
    128,
    Some(0.0),
    Some(1.0),
    Deterministic::new(9),
  )
  .sample();
  let got_seeded = merton_base_seeded(Deterministic::new(9))
    .with_points(128)
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn merton_with_horizon_matches_fresh_construction() {
  let mut expected = merton_base_seeded(Unseeded);
  expected.t = Some(2.0);
  let got = merton_base_seeded(Unseeded).with_horizon(Some(2.0));
  assert_eq!(got.t, Some(2.0));
  assert_eq!(fields(&got), fields(&expected));

  let want = Merton::new(
    0.03,
    0.2,
    1.0,
    0.0,
    SimdNormal::new(0.0, 0.1),
    64,
    Some(0.0),
    Some(2.0),
    Deterministic::new(11),
  )
  .sample();
  let got_seeded = merton_base_seeded(Deterministic::new(11))
    .with_horizon(Some(2.0))
    .sample();
  assert_eq!(want, got_seeded);
}

#[test]
fn merton_with_seed_matches_fresh_construction() {
  let want = Merton::new(
    0.03,
    0.2,
    1.0,
    0.0,
    SimdNormal::new(0.0, 0.1),
    64,
    Some(0.0),
    Some(1.0),
    Deterministic::new(13),
  )
  .sample();
  let got = merton_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}

/// The round trip rooted in `Default` rather than in `merton_base()`.
#[test]
fn merton_default_with_alpha_round_trip() {
  let base = Merton::<f64, SimdNormal<f64>>::default();
  let got = Merton::<f64, SimdNormal<f64>>::default().with_alpha(0.06);
  let expected = MertonFields {
    alpha: 0.06,
    ..fields(&base)
  };
  assert_eq!(got.alpha, 0.06);
  assert_eq!(fields(&got), expected);
  assert!(finite(&got.sample()));
}
