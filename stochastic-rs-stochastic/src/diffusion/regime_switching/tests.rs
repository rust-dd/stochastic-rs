use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

fn q3() -> Array2<f64> {
  array![[-1.0, 0.5, 0.5], [0.5, -1.0, 0.5], [0.5, 0.5, -1.0],]
}

#[test]
fn price_stays_positive() {
  let p = RegimeSwitchingDiffusion::new(
    0.05,
    q3(),
    array![0.15, 0.25, 0.35],
    0,
    1000,
    100.0,
    Some(1.0),
    Unseeded,
  );
  let [s, _z] = p.sample();
  assert!(s.iter().all(|x| *x > 0.0));
}

#[test]
fn regime_states_valid() {
  let p = RegimeSwitchingDiffusion::new(
    0.05,
    q3(),
    array![0.15, 0.25, 0.35],
    0,
    1000,
    100.0,
    Some(1.0),
    Unseeded,
  );
  let [_s, z] = p.sample();
  for &state in z.iter() {
    assert!((0.0..3.0).contains(&state), "invalid state={state}");
  }
}

#[test]
fn single_regime_like_gbm() {
  let q1 = array![[0.0]];
  let p = RegimeSwitchingDiffusion::new(
    0.05,
    q1,
    array![0.2],
    0,
    1000,
    100.0,
    Some(1.0),
    Deterministic::new(42),
  );
  let [s, z] = p.sample();
  assert!(s.iter().all(|x| *x > 0.0));
  assert!(z.iter().all(|x| (*x - 0.0_f64).abs() < 1e-10));
}

#[test]
fn seeded_is_deterministic() {
  let p1 = RegimeSwitchingDiffusion::new(
    0.05,
    q3(),
    array![0.15, 0.25, 0.35],
    0,
    100,
    100.0,
    Some(1.0),
    Deterministic::new(42),
  );
  let p2 = RegimeSwitchingDiffusion::new(
    0.05,
    q3(),
    array![0.15, 0.25, 0.35],
    0,
    100,
    100.0,
    Some(1.0),
    Deterministic::new(42),
  );
  let [s1, z1] = p1.sample();
  let [s2, z2] = p2.sample();
  assert_eq!(s1, s2);
  assert_eq!(z1, z2);
}

#[test]
fn matrix_exp_identity() {
  let zero = Array2::<f64>::zeros((3, 3));
  let result = matrix_exp_real(&zero);
  for i in 0..3 {
    for j in 0..3 {
      let expected = if i == j { 1.0 } else { 0.0 };
      assert!(
        (result[[i, j]] - expected).abs() < 1e-10,
        "exp(0)[{i}][{j}]={}, expected {expected}",
        result[[i, j]]
      );
    }
  }
}

#[test]
fn transition_probs_sum_to_one() {
  let p = RegimeSwitchingDiffusion::new(
    0.05,
    q3(),
    array![0.15, 0.25, 0.35],
    0,
    10,
    100.0,
    Some(1.0),
    Unseeded,
  );
  let pm = p.transition_prob_matrix(0.01_f64);
  for i in 0..3 {
    let sum: f64 = (0..3).map(|j| pm[[i, j]]).sum();
    assert!((sum - 1.0).abs() < 1e-10, "row sum={sum}");
  }
}
