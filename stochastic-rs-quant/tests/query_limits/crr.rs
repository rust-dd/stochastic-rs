//! The Cox–Ross–Rubinstein lattice: its probability rule, a NaN query and the tree's own check.

use ndarray::array;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::lattice::BinomialTree;
use stochastic_rs_quant::lattice::CrrModel;
use stochastic_rs_quant::traits::ModelPricer;

const K: f64 = 100.0;
const R: f64 = 0.05;
const Q: f64 = 0.02;
const SPOTS: [f64; 3] = [90.0, 100.0, 110.0];
const TYPES: [OptionType; 2] = [OptionType::Call, OptionType::Put];

/// CRR (1979): `p = (e^{(r−q)Δt} − d)/(u − d)` is a probability only inside `[0, 1]`.
#[test]
fn the_crr_lattice_answers_nan_outside_the_unit_interval_and_for_a_nan_query() {
  // dt = 0.5, u = e^{0.2 sqrt(0.5)} = 1.152, d = 0.868, growth = e^{0.25} = 1.284: p = 1.46.
  let coarse = CrrModel::new(0.2, 2);
  assert!(coarse.price_call(100.0, 100.0, 0.5, 0.0, 1.0).is_nan());
  assert!(coarse.price_put(100.0, 100.0, 0.5, 0.0, 1.0).is_nan());
  for option_type in TYPES {
    assert!(
      coarse
        .price_american(100.0, 100.0, 0.5, 0.0, 1.0, option_type)
        .is_nan()
    );
    assert!(
      coarse
        .price_american(100.0, 100.0, -0.5, 0.0, 1.0, option_type)
        .is_nan()
    );
  }
  let fine = CrrModel::new(0.2, 400);
  assert!(fine.price_call(100.0, 100.0, 0.5, 0.0, 1.0).is_finite());
  let nan = f64::NAN;
  for (s, k, r, q, tau) in [
    (nan, K, R, Q, 1.0),
    (90.0, nan, R, Q, 1.0),
    (90.0, K, nan, Q, 1.0),
    (90.0, K, R, nan, 1.0),
    (nan, K, R, Q, 0.0),
    (90.0, nan, R, Q, 0.0),
  ] {
    for option_type in TYPES {
      let american = fine.price_american(s, k, r, q, tau, option_type);
      let european = fine.price_option(s, k, r, q, tau, option_type);
      assert!(
        american.is_nan() && european.is_nan(),
        "{s} {k} {r} {q} {tau}: {american}"
      );
    }
  }
  assert!(
    CrrModel::new(0.0, 50)
      .price_call(100.0, K, R, Q, 1.0)
      .is_nan()
  );
}

/// Reference: the Black–Scholes limit of CRR (1979) §5, and the American put above the European.
#[test]
fn the_crr_lattice_inside_its_domain_keeps_its_prices() {
  use stochastic_rs_quant::pricing::bsm::BsmCoc;
  use stochastic_rs_quant::pricing::bsm::BsmPricer;
  let crr = CrrModel::new(0.2, 500);
  let bsm = BsmPricer::new(0.2, BsmCoc::Merton1973);
  for s in SPOTS {
    for option_type in TYPES {
      let (tree, closed) = (
        crr.price_option(s, K, R, Q, 1.0, option_type),
        bsm.price_option(s, K, R, Q, 1.0, option_type),
      );
      assert!(
        (tree - closed).abs() < 1e-2,
        "{option_type:?} at {s}: {tree} vs {closed}"
      );
    }
    let american = crr.price_american(s, K, R, Q, 1.0, OptionType::Put);
    assert!(american >= crr.price_put(s, K, R, Q, 1.0) && american >= K - s);
  }
}

#[test]
#[should_panic(
  expected = "up_probabilities[1][0] must satisfy `p.is_finite() && p >= T::zero() && p <= T::one()`, got up_probabilities[1][0] = 1.5"
)]
fn a_binomial_tree_with_a_probability_above_one_is_a_precondition() {
  BinomialTree::new(
    0.5,
    vec![
      array![100.0],
      array![90.0, 110.0],
      array![81.0, 99.0, 121.0],
    ],
    vec![array![0.5], array![1.5, 0.5]],
  );
}

#[test]
#[should_panic(
  expected = "up_probabilities[0][0] must satisfy `p.is_finite() && p >= T::zero() && p <= T::one()`, got up_probabilities[0][0] = NaN"
)]
fn a_binomial_tree_from_a_nan_probability_is_a_precondition() {
  BinomialTree::from_crr(100.0, 1.1, 1.0 / 1.1, f64::NAN, 2, 0.5);
}

#[test]
#[should_panic(expected = "steps must satisfy `steps >= 1`, got steps = 0")]
fn a_crr_lattice_without_a_step_is_a_precondition() {
  CrrModel::new(0.2, 0);
}

#[test]
#[should_panic(
  expected = "sigma must satisfy `sigma.is_finite() && sigma >= T::zero()`, got sigma = -0.2"
)]
fn a_crr_lattice_with_a_negative_volatility_is_a_precondition() {
  CrrModel::new(-0.2, 50);
}

#[test]
#[should_panic(
  expected = "sigma must satisfy `sigma.is_finite() && sigma >= T::zero()`, got sigma = NaN"
)]
fn a_crr_lattice_with_a_nan_volatility_is_a_precondition() {
  CrrModel::new(f64::NAN, 50);
}

#[test]
#[should_panic(
  expected = "sigma must satisfy `sigma.is_finite() && sigma >= T::zero()`, got sigma = inf"
)]
fn a_crr_lattice_with_an_infinite_volatility_is_a_precondition() {
  CrrModel::new(f64::INFINITY, 50);
}

/// The lattice holds no rate: an infinite one arrives with the query and answers NaN.
#[test]
fn a_crr_lattice_is_nan_for_an_infinite_rate() {
  let crr = CrrModel::new(0.2, 50);
  for (r, q) in [
    (f64::INFINITY, Q),
    (R, f64::INFINITY),
    (f64::NEG_INFINITY, Q),
  ] {
    for option_type in TYPES {
      let european = crr.price_option(100.0, K, r, q, 1.0, option_type);
      let american = crr.price_american(100.0, K, r, q, 1.0, option_type);
      assert!(
        european.is_nan() && american.is_nan(),
        "{r} {q}: {european} {american}"
      );
    }
  }
}
