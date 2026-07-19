use ndarray::array;

use super::*;

#[test]
fn quarticity_term_has_physical_period_scaling() {
  let quarticity = 1.75;
  let n_freq = 4;
  let unit = quarticity_mse_term(1.0, quarticity, n_freq);
  let paper = quarticity_mse_term(std::f64::consts::TAU, quarticity, n_freq);

  assert_eq!(unit, 4.0 * quarticity / 9.0);
  assert_eq!(paper, 8.0 * std::f64::consts::PI * quarticity / 9.0);
}

#[test]
fn uniform_mean_dirichlet_is_the_regular_grid_kernel() {
  let durations = Array1::from_elem(8, 0.125);
  let expected = rescaled_dirichlet_kernel(3, 0.125, 1.0);
  let actual = mean_rescaled_dirichlet_kernel(3, &durations, 1.0);

  assert_eq!(actual, expected);
}

/// Hand-evaluated regression for the documented irregular-grid heuristic.
/// The expected values follow directly from the seven listed increments and
/// durations, before invoking any production helper.
#[test]
fn irregular_grid_heuristic_matches_fixed_mse_golden() {
  let prices = array![0.0, 0.11, -0.03, 0.18, 0.07, 0.22, 0.16, 0.31];
  let irregular = array![0.0, 0.08, 0.24, 0.39, 0.61, 0.74, 0.93, 1.0];
  let result =
    try_optimal_cutting_frequency(prices.as_slice().unwrap(), irregular.as_slice().unwrap())
      .unwrap();
  let expected = [
    0.017_012_273_695_047_37,
    0.013_917_181_100_193_216,
    0.014_203_381_424_272_748,
  ];

  assert_eq!(result.n_opt, 2);
  assert_eq!(result.noise_variance, 0.0);
  for (actual, expected) in result.mse_curve.iter().zip(expected) {
    assert!((actual - expected).abs() < 1e-15);
  }
}

#[test]
fn optimal_cutting_rejects_invalid_timestamps() {
  let prices = Array1::linspace(0.0, 0.7, 8);
  let duplicate = array![0.0, 0.1, 0.2, 0.2, 0.5, 0.7, 0.8, 1.0];
  let non_finite = array![0.0, 0.1, 0.2, 0.3, f64::NAN, 0.7, 0.8, 1.0];

  assert!(
    try_optimal_cutting_frequency(prices.as_slice().unwrap(), duplicate.as_slice().unwrap())
      .is_err()
  );
  assert!(
    try_optimal_cutting_frequency(prices.as_slice().unwrap(), non_finite.as_slice().unwrap())
      .is_err()
  );
}
