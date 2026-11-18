use std::convert::Infallible;

use super::*;

struct Sphere;

impl CostFunction for Sphere {
  type Param = Vec<f64>;
  type Output = f64;
  type Error = Infallible;

  fn cost(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
    Ok(x.iter().map(|value| value.powi(2)).sum())
  }
}

#[test]
fn simplex_spread_uses_sample_standard_deviation() {
  assert_eq!(sample_standard_deviation(&[1.0, 2.0, 3.0]), Some(1.0));
  assert_eq!(sample_standard_deviation(&[0.0, 0.0]), Some(0.0));
  assert_eq!(sample_standard_deviation(&[f64::MAX, f64::MAX]), Some(0.0));
}

#[test]
fn undefined_simplex_spread_is_rejected() {
  for costs in [&[][..], &[1.0], &[0.0, f64::INFINITY], &[0.0, f64::NAN]] {
    assert_eq!(sample_standard_deviation(costs), None);
  }
}

#[test]
#[should_panic(expected = "simplex must contain at least two vertices")]
fn empty_simplex_has_an_explicit_precondition() {
  run_nelder_mead(Sphere, vec![], 100, 1e-10, false);
}

#[test]
fn invalid_simplex_tolerance_returns_the_initial_point() {
  let initial = vec![2.0];
  for tolerance in [-1.0, f64::NAN, f64::INFINITY] {
    let simplex = vec![initial.clone(), vec![3.0]];
    let run = run_nelder_mead(Sphere, simplex, 100, tolerance, true);
    assert_eq!(run.best, initial);
    assert!(!run.converged);
    assert_eq!((run.iterations, run.trail), (0, vec![initial.clone()]));
  }
}

#[test]
fn a_converged_run_reports_the_iterations_it_took() {
  let simplex = vec![vec![2.0], vec![3.0]];

  let run = run_nelder_mead(Sphere, simplex.clone(), 400, 1e-10, true);
  let capped = run_nelder_mead(Sphere, simplex, run.iterations - 1, 1e-10, false);

  assert!(run.converged && run.best[0].abs() < 1e-4 && run.iterations > 1);
  assert_eq!(capped.iterations, run.iterations - 1);
  assert!(capped.trail.is_empty());
  assert_eq!(run.trail.len() as u64, run.iterations + 1);
  assert_eq!(run.trail.last(), Some(&run.best));
}

#[test]
fn a_nelder_mead_trail_records_market_minus_model() {
  let options = CalibrationOptions::default().with_record_history(true);
  let simplex = vec![vec![2.0], vec![3.0]];
  let fit = nelder_mead_fit(Sphere, simplex, &options, 400, |x| {
    (x[0], vec![1.0], vec![1.0 + x[0]])
  })
  .unwrap();
  assert!(fit.history.len() > 1);
  for entry in &fit.history {
    assert_eq!(entry.residuals[0], 1.0 - (1.0 + entry.params));
  }
}

/// McKinnon's function with `(τ, θ, φ)`, from Kelley's (1999, §4) simplex. At (2, 6, 60) plain
/// Nelder–Mead contracts onto the origin, where ∂f/∂y = 1; the minimum is (0, −1/2).
struct McKinnon(f64, f64, f64);

const SMOOTH: McKinnon = McKinnon(2.0, 6.0, 60.0);

impl CostFunction for McKinnon {
  type Param = Vec<f64>;
  type Output = f64;
  type Error = Infallible;

  fn cost(&self, p: &Self::Param) -> Result<f64, Infallible> {
    let Self(tau, theta, phi) = *self;
    let (x, y) = (p[0], p[1]);
    let scale = if x <= 0.0 { theta * phi } else { theta };
    Ok(scale * x.abs().powf(tau) + y + y * y)
  }
}

fn mckinnon_simplex() -> Vec<Vec<f64>> {
  let lambda = |sign: f64| (1.0 + sign * 33f64.sqrt()) / 8.0;
  vec![
    vec![1.0, 1.0],
    vec![lambda(1.0), lambda(-1.0)],
    vec![0.0, 0.0],
  ]
}

#[test]
fn an_oriented_restart_leaves_mckinnons_stagnation_point() {
  let run = run_nelder_mead(SMOOTH, mckinnon_simplex(), 500, 1e-8, false);
  assert!(run.converged, "{:?} after {}", run.best, run.iterations);
  assert!(
    run.best[0].abs() < 1e-3 && (run.best[1] + 0.5).abs() < 1e-3,
    "{:?}",
    run.best
  );
}

#[test]
fn a_simplex_without_volume_is_not_convergence() {
  let options = CalibrationOptions::default();
  let simplex = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
  let fit = nelder_mead_fit(SMOOTH, simplex, &options, 400, |x| {
    (x.to_vec(), vec![0.0], vec![x[0]])
  })
  .unwrap();
  assert!(!fit.converged);
  assert_eq!(fit.message.as_deref(), Some("simplex collapsed"));
}

#[test]
fn a_nelder_mead_fit_names_its_stop() {
  let simplex = vec![vec![2.0], vec![3.0]];
  let price = |x: &[f64]| (x[0], vec![1.0], vec![1.0 + x[0]]);
  let options = CalibrationOptions::default();
  let fit = nelder_mead_fit(Sphere, simplex.clone(), &options, 400, price).unwrap();
  assert!(fit.converged);
  assert_eq!(fit.message.as_deref(), Some("converged"));
  let capped = nelder_mead_fit(Sphere, simplex, &options, 3, price).unwrap();
  assert!(!capped.converged);
  assert_eq!(capped.message.as_deref(), Some("iteration cap reached"));
}

#[test]
fn iteration_limit_is_not_reported_as_convergence() {
  let simplex = vec![vec![2.0], vec![3.0]];

  let run = run_nelder_mead(Sphere, simplex, 0, 1e-10, false);

  assert!(!run.converged);
  assert_eq!(run.iterations, 0);
}

#[test]
fn nonfinite_simplex_costs_stop_without_convergence() {
  struct Nonfinite;

  impl CostFunction for Nonfinite {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Infallible> {
      Ok(if x[0] > 0.0 { f64::INFINITY } else { 0.0 })
    }
  }

  let state = basin::BasicSimplexState::from_simplex(vec![vec![0.0], vec![1.0]]);
  let result = basin::Executor::new(Nonfinite, basin::NelderMead::new(), state)
    .max_iter(100)
    .stop_when(|state| {
      super::SimplexStandardDeviation::new(1e-10, vec![1.0])
        .unwrap()
        .check(state)
    })
    .run()
    .unwrap();
  assert_eq!(result.reason, basin::TerminationReason::SolverFailed);
  assert_eq!(result.iter(), 0);
  assert_eq!(result.best_param(), &vec![0.0]);
}

/// Kelley's §4 case (τ, θ, φ) = (1, 15, 10), not differentiable at the stagnation point: his run
/// ends on the third restart.
#[test]
fn a_third_missed_decrease_is_stagnation() {
  let run = run_nelder_mead(
    McKinnon(1.0, 15.0, 10.0),
    mckinnon_simplex(),
    500,
    1e-8,
    false,
  );
  assert_eq!(run.stop, SimplexStop::Stagnated, "{:?}", run.best);
  assert!(!run.converged && run.iterations < 100, "{}", run.iterations);
}

/// Three vertices on the unit circle cost the same, a zero spread on a simplex that has not
/// contracted.
#[test]
fn equal_costs_on_a_simplex_that_has_not_contracted_are_not_a_stop() {
  let simplex = vec![vec![1.0, 0.0], vec![-1.0, 0.0], vec![0.0, 1.0]];
  let run = run_nelder_mead(Sphere, simplex, 400, 1e-10, false);
  assert!(run.converged, "{:?}", run.stop);
  assert!(run.best.iter().all(|x| x.abs() < 1e-4), "{:?}", run.best);
}

struct Constant;

impl CostFunction for Constant {
  type Param = Vec<f64>;
  type Output = f64;
  type Error = Infallible;

  fn cost(&self, _: &Self::Param) -> Result<f64, Infallible> {
    Ok(3.0)
  }
}

#[test]
fn a_cost_no_parameter_moves_is_flat_not_converged() {
  let simplex = vec![vec![1.0, 1.0], vec![2.0, 1.0], vec![1.0, 2.0]];
  let run = run_nelder_mead(Constant, simplex, 400, 1e-10, false);
  assert!(!run.converged);
  assert_eq!(run.stop.message(), "flat objective");
}

/// Flat left of `x = 5e-4`, a bowl around (1, 0) right of it: the simplex shrinks onto the
/// origin, and only a point `0.001·step` away is lower.
struct Shelf;

impl CostFunction for Shelf {
  type Param = Vec<f64>;
  type Output = f64;
  type Error = Infallible;

  fn cost(&self, p: &Self::Param) -> Result<f64, Infallible> {
    Ok(if p[0] < 5e-4 {
      1.0
    } else {
      (p[0] - 1.0).powi(2) + p[1].powi(2)
    })
  }
}

#[test]
fn a_lower_point_beside_a_spread_stop_restarts_the_fit() {
  let simplex = vec![vec![0.0, 0.0], vec![-1.0, 0.0], vec![0.0, -1.0]];
  let run = run_nelder_mead(Shelf, simplex, 400, 1e-10, false);
  assert!(run.converged, "{:?} at {:?}", run.stop, run.best);
  assert!(
    (run.best[0] - 1.0).abs() < 1e-4 && run.best[1].abs() < 1e-4,
    "{:?}",
    run.best
  );
}
