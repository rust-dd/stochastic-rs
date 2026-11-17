use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::ou::Ou;

use super::*;
use crate::qmle::fit_ou_mle;
use crate::qmle::test_support::simulate_ou;

/// From a level and a `σ` far off, the search slides to a speed near its bound, where the level
/// has no gradient: run again from the start's speed and level with `σ` fitted, it reaches the
/// closed form's optimum.
#[test]
fn a_far_start_that_stalls_where_the_level_is_not_identified_reaches_the_closed_form() {
  let dt = 1.0 / 252.0;
  for (level, seed) in [(1.0, 13u64), (0.03, 2), (-0.2, 2)] {
    let path = simulate_ou(8.0, level, 0.02, level, dt, 1000, seed);
    let closed = fit_ou_mle(path.view(), dt).unwrap();
    let mut model = Ou::new(
      1.0,
      0.5,
      0.5,
      100,
      Some(1.0),
      Some(1.0),
      Deterministic::new(0),
    );
    let fit = fit_mle(&mut model, path.view(), dt, DensityApprox::Exact, None).unwrap();
    assert!(fit.converged, "level {level}");
    assert!(
      (fit.log_likelihood - closed.log_likelihood).abs() < 1e-6,
      "level {level}: {} for {} at {}",
      fit.log_likelihood,
      closed.log_likelihood,
      fit.params
    );
  }
}

/// A parameter whose move alone to an end of its box costs under 1.92 goes back to the start's
/// value, and one the likelihood holds stays; a fixed one, with no room, stays too.
#[test]
fn only_a_parameter_the_likelihood_does_not_hold_goes_back_to_its_start() {
  let dt = 1.0 / 252.0;
  let path = simulate_ou(8.0, 1.0, 0.02, 1.0, dt, 1000, 13);
  let loosened = |lower: [f64; 3], upper: [f64; 3], at: [f64; 3]| {
    let mut model = Ou::new(1.0, 0.5, 0.5, 100, None, None, Deterministic::new(0));
    let problem = MleProblem {
      model: Mutex::new(&mut model),
      sample: path.view(),
      dt,
      density: DensityApprox::Exact,
      lower: lower.to_vec(),
      upper: upper.to_vec(),
    };
    problem.loosened(&at, &[1.0, 0.5, 0.5])
  };
  let (lower, upper) = ([1e-4, -10.0, 1e-6], [50.0, 10.0, 10.0]);
  let stalled = [1e-4, 8.35, 0.0188];
  assert_eq!(loosened(lower, upper, stalled), [1.0, 0.5, 0.0188]);
  let closed = fit_ou_mle(path.view(), dt).unwrap();
  let optimum = [closed.theta, closed.mu, closed.sigma];
  assert_eq!(loosened(lower, upper, optimum), optimum);
  let fixed = loosened([1e-4, 8.35, 1e-6], [50.0, 8.35, 10.0], stalled);
  assert_eq!(fixed, [1.0, 8.35, 0.0188]);
}
