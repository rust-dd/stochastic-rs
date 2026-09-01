// docs: stats#lee-mykland-jump-test
//! Backs the Lee–Mykland example on the stats page.

use ndarray::Array1;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stats::realized::lee_mykland::lee_mykland_test;
use stochastic_rs::stats::realized::lee_mykland::lee_mykland_window;

#[test]
fn lee_mykland_locates_a_planted_jump() {
  // Stand-in for a day of 5-minute log-returns with one news jump in it.
  let dist = SimdNormal::<f64>::new(0.0, 0.001, &Deterministic::new(42));
  let mut returns = Array1::<f64>::zeros(2_000);
  dist.fill_slice(returns.as_slice_mut().unwrap());
  returns[1_200] = 0.02;

  // The paper's window rule for 78 five-minute returns a day is K = 141.
  let window = lee_mykland_window(78);
  assert_eq!(window, 141);

  let test = lee_mykland_test(returns.view(), window, 0.01);
  assert!(test.jump_indices.contains(&1_200));
  // Every flagged return sits beyond the Gumbel threshold on |L(i)|.
  for &i in &test.jump_indices {
    assert!(test.statistics[i].abs() > test.threshold);
  }
}
