use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::ou::Ou;

use super::*;
use crate::qmle::fit_ou_mle;
use crate::qmle::test_support::simulate_ou;

/// From a level and a `σ` far off the search slides to a speed near its bound, where the level has
/// no gradient: again from the start's speed and level with `σ` fitted, it reaches the closed form.
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

/// A level in an open box has no end to be loose at: the start's value stands in for it, so the
/// search that slid to a speed of `1e-4` and a level of `−530` times the path's runs again.
#[test]
fn a_stall_in_an_open_box_is_searched_again_from_the_start() {
  let dt = 1.0 / 252.0;
  let c = 1e9;
  let path = simulate_ou(2.0, 1.0, 0.3, 1.0, dt, 1000, 102).mapv(|v| c * v);
  let closed = fit_ou_mle(path.view(), dt).unwrap();
  let open = (f64::NEG_INFINITY, f64::INFINITY);
  let bounds = vec![(1e-4, 50.0), open, (1e-9, 100.0 * c)];
  let mut model = Ou::new(1.0, c, 1e-3 * c, 100, None, None, Deterministic::new(0));
  let exact = DensityApprox::Exact;
  let fit = fit_mle(&mut model, path.view(), dt, exact, Some(bounds)).unwrap();
  let short = closed.log_likelihood - fit.log_likelihood;
  assert!(
    fit.converged && short.abs() < 1e-3,
    "{short} below at {}",
    fit.params
  );
}

/// A parameter whose move alone to an end of its box costs under 1.92 goes back to the start's
/// value, and one the likelihood holds stays; with the level fixed one end is loose and none moves.
#[test]
fn only_a_parameter_the_likelihood_does_not_hold_goes_back_to_its_start() {
  let dt = 1.0 / 252.0;
  let path = simulate_ou(8.0, 1.0, 0.02, 1.0, dt, 1000, 13);
  let loosened = |lower: [f64; 3], upper: [f64; 3], at: [f64; 3]| {
    let mut model = Ou::new(1.0, 0.5, 0.5, 100, None, None, Deterministic::new(0));
    let exact = DensityApprox::Exact;
    let start = [1.0, 0.5, 0.5];
    let problem = MleProblem::new(&mut model, path.view(), dt, exact, &lower, &upper, &start);
    problem.loosened(&at, &start)
  };
  let (lower, upper) = ([1e-4, -10.0, 1e-6], [50.0, 10.0, 10.0]);
  let stalled = [1e-4, 8.35, 0.0188];
  assert_eq!(loosened(lower, upper, stalled), [1.0, 0.5, 0.0188]);
  let closed = fit_ou_mle(path.view(), dt).unwrap();
  let optimum = [closed.theta, closed.mu, closed.sigma];
  assert_eq!(loosened(lower, upper, optimum), optimum);
  let fixed = loosened([1e-4, 8.35, 1e-6], [50.0, 8.35, 10.0], stalled);
  assert_eq!(fixed, stalled);
}

/// From the law's parameters in a box with three infinite ends the estimate stays by its start,
/// which then stands in for no end: the fit is its first search, to the iteration.
#[test]
fn a_fit_beside_its_start_in_an_open_box_runs_no_further_search() {
  let dt = 1.0 / 252.0;
  let inf = f64::INFINITY;
  let (lower, upper, start) = ([1e-4, -inf, 1e-9], [inf, inf, inf], [2.0, 1.0, 0.3]);
  let exact = DensityApprox::Exact;
  for seed in 101..=103 {
    let path = simulate_ou(2.0, 1.0, 0.3, 1.0, dt, 1000, seed);
    let mut model = Ou::new(2.0, 1.0, 0.3, 100, None, None, Deterministic::new(0));
    let problem = MleProblem::new(&mut model, path.view(), dt, exact, &lower, &upper, &start);
    let first = search(problem, &start);
    let bounds = lower.into_iter().zip(upper).collect::<Vec<_>>();
    let mut model = Ou::new(2.0, 1.0, 0.3, 100, None, None, Deterministic::new(0));
    let fit = fit_mle(&mut model, path.view(), dt, exact, Some(bounds)).unwrap();
    assert!(fit.converged, "seed {seed}: {}", fit.message);
    assert_eq!(fit.params.to_vec(), first.params, "seed {seed}");
    assert_eq!(fit.iterations, first.iterations, "seed {seed}");
  }
}

/// 50 points do not hold the speed at its lower end, and that one loose end is an optimum beside
/// a bound, not a stall: no further search runs, and `iterations` is the first search's.
#[test]
fn a_fit_with_one_loose_end_runs_no_further_search() {
  let dt = 1.0 / 252.0;
  let path = simulate_ou(2.0, 1.0, 0.3, 1.0, dt, 50, 1);
  let (lower, upper, start) = ([1e-4, -10.0, 1e-6], [50.0, 10.0, 10.0], [2.0, 1.0, 0.3]);
  let exact = DensityApprox::Exact;
  let mut model = Ou::new(2.0, 1.0, 0.3, 100, None, None, Deterministic::new(0));
  let problem = MleProblem::new(&mut model, path.view(), dt, exact, &lower, &upper, &start);
  let first = search(problem, &start);
  let problem = MleProblem::new(&mut model, path.view(), dt, exact, &lower, &upper, &start);
  let ends = |i: usize, end: f64| {
    let mut moved = first.params.clone();
    moved[i] = end;
    problem.eval_nll(&moved) - first.cost < 1.92
  };
  let loose = [
    ends(0, 1e-4),
    ends(0, 50.0),
    ends(1, -10.0),
    ends(1, 10.0),
    ends(2, 1e-6),
    ends(2, 10.0),
  ];
  assert_eq!(loose, [true, false, false, false, false, false]);
  drop(problem);
  let mut model = Ou::new(2.0, 1.0, 0.3, 100, None, None, Deterministic::new(0));
  let fit = fit_mle(&mut model, path.view(), dt, exact, None).unwrap();
  assert_eq!(fit.params.to_vec(), first.params);
  assert_eq!(fit.iterations, first.iterations);
}

/// From a `σ` twelve orders below the estimate the first search ends far from its units and runs
/// again in new ones: `iterations` sums every search, within 16 of 200.
#[test]
fn iterations_count_the_searches_in_new_units() {
  let dt = 1.0 / 252.0;
  let path = simulate_ou(2.0, 1.0, 0.3, 1.0, dt, 1000, 101).mapv(|v| 1e9 * v);
  let (lower, upper, start) = ([1e-4, -1e11, 1e-9], [50.0, 1e11, 1e11], [1.0, 1e9, 1e-6]);
  let exact = DensityApprox::Exact;
  let mut model = Ou::new(1.0, 1e9, 1e-6, 100, None, None, Deterministic::new(0));
  let problem = MleProblem::new(&mut model, path.view(), dt, exact, &lower, &upper, &start);
  let first = search(problem, &start);
  let bounds = lower.into_iter().zip(upper).collect::<Vec<_>>();
  let mut model = Ou::new(1.0, 1e9, 1e-6, 100, None, None, Deterministic::new(0));
  let fit = fit_mle(&mut model, path.view(), dt, exact, Some(bounds)).unwrap();
  assert!(
    first.params[2] > 8.0 * start[2],
    "σ ended at {}",
    first.params[2]
  );
  assert!(
    first.iterations < fit.iterations && fit.iterations <= 16 * 200,
    "{} for the first search's {}",
    fit.iterations,
    first.iterations
  );
}
