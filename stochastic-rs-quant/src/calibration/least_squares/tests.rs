use super::*;

struct Rosenbrock {
  x: Array1<f64>,
  calls: Cell<usize>,
  fail_at: Option<usize>,
  reject_outside_domain: bool,
  nonfinite_trials: Cell<usize>,
  accepted: Vec<Array1<f64>>,
}

impl Rosenbrock {
  fn new() -> Self {
    Self {
      x: Array1::from_vec(vec![-1.2, 1.0]),
      calls: Cell::new(0),
      fail_at: None,
      reject_outside_domain: false,
      nonfinite_trials: Cell::new(0),
      accepted: Vec::new(),
    }
  }

  fn cost(x: &Array1<f64>) -> f64 {
    50.0 * (x[1] - x[0].powi(2)).powi(2) + 0.5 * (1.0 - x[0]).powi(2)
  }
}

impl LeastSquaresProblem for Rosenbrock {
  fn set_params(&mut self, x: &Array1<f64>) {
    self.x.clone_from(x);
  }

  fn params(&self) -> Array1<f64> {
    self.x.clone()
  }

  fn residuals(&mut self) -> Option<Array1<f64>> {
    self.calls.set(self.calls.get() + 1);
    if self.fail_at == Some(self.calls.get()) {
      return None;
    }
    if self.reject_outside_domain && self.x[1] < -2.0 {
      self.nonfinite_trials.set(self.nonfinite_trials.get() + 1);
      return Some(Array1::from_elem(2, f64::NAN));
    }
    Some(Array1::from_vec(vec![
      10.0 * (self.x[1] - self.x[0].powi(2)),
      1.0 - self.x[0],
    ]))
  }

  fn jacobian(&self) -> Option<Array2<f64>> {
    Array2::from_shape_vec((2, 2), vec![-20.0 * self.x[0], 10.0, -1.0, 0.0]).ok()
  }

  fn accept(&mut self) {
    self.accepted.push(self.x.clone());
  }
}

#[test]
fn the_start_and_each_trial_that_lowers_the_cost_are_accepted() {
  for pivoted_qr in [false, true] {
    let mut problem = Rosenbrock::new();
    problem.reject_outside_domain = true;
    let (problem, report) = minimize(
      problem,
      LmOptions {
        pivoted_qr,
        ..LmOptions::default()
      },
    );
    let accepted = &problem.accepted;
    assert_eq!(accepted[0], Rosenbrock::new().x);
    assert!(problem.nonfinite_trials.get() > 0 && accepted.iter().all(|x| x[1] >= -2.0));
    assert!(accepted.len() < report.evaluations);
    assert!(
      accepted
        .windows(2)
        .all(|pair| Rosenbrock::cost(&pair[1]) < Rosenbrock::cost(&pair[0]))
    );
    assert_eq!(accepted.last(), Some(&problem.x));
  }
}

#[test]
fn rosenbrock_converges_with_consistent_evaluation_counts() {
  for pivoted_qr in [false, true] {
    let (problem, report) = minimize(
      Rosenbrock::new(),
      LmOptions {
        pivoted_qr,
        ..LmOptions::default()
      },
    );
    assert!(report.converged, "{report:?}");
    assert_eq!(report.message(), "converged");
    assert!(distance_to_optimum(&problem.x) < 1e-8);
    assert_eq!(report.evaluations, problem.calls.get());
  }
}

#[test]
fn the_report_builds_the_calibrated_fit() {
  use crate::traits::CalibrationResult;
  let (problem, report) = minimize(Rosenbrock::new(), LmOptions::default());
  let fit = report
    .calibrated(
      problem.x.to_vec(),
      &[1.0, 2.0],
      &[1.5, 2.0],
      &LossMetric::ALL,
      vec![],
    )
    .unwrap();
  assert!(fit.converged() && fit.history().is_empty());
  assert_eq!(fit.iterations(), Some(report.iterations));
  assert_eq!(fit.message(), Some("converged"));
  assert_eq!(fit.max_error(), Some(0.5));
  assert_eq!(fit.rmse(), 0.125_f64.sqrt());
}

#[test]
fn a_trail_keeps_the_last_staged_point_once_per_accept_and_numbers_it() {
  let options = CalibrationOptions::default().with_record_history(true);
  let (market, rows) = ([1.0, 2.0], Array1::from_vec(vec![0.5, -1.0]));
  let mut trail = Trail::new();
  trail.stage(&options, &market, &[0.5, 3.0], &0.1, &rows);
  trail.accept();
  trail.stage(&options, &market, &[9.0, 9.0], &0.2, &rows);
  trail.stage(&options, &market, &[1.0, 2.0], &0.3, &rows);
  trail.accept();
  trail.accept();
  let entries = trail.into_entries();
  let numbered = entries.iter().map(|e| (e.iteration, e.params));
  assert_eq!(numbered.collect::<Vec<_>>(), vec![(0, 0.1), (1, 0.3)]);
  assert!(entries.iter().all(|entry| entry.residuals == rows));
  assert_eq!(entries[0].loss.get(LossMetric::Mae), 0.75);
  assert_eq!(entries[1].loss.get(LossMetric::Rmse), 0.0);
}

#[test]
fn a_negated_trail_stores_the_rows_negated_and_an_unrecorded_one_nothing() {
  let rows = Array1::from_vec(vec![0.5, -0.0, f64::NAN]);
  let mut recorded = Trail::negated();
  let options = CalibrationOptions::default().with_record_history(true);
  recorded.stage(&options, &[1.0; 3], &[1.0; 3], &(), &rows);
  recorded.accept();
  let stored = &recorded.into_entries()[0].residuals;
  let bits = |v: &Array1<f64>| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
  assert_eq!(bits(stored), bits(&rows.mapv(|x| -x)));
  let mut unrecorded = Trail::negated();
  unrecorded.stage(&CalibrationOptions::default(), &[1.0], &[2.0], &(), &rows);
  unrecorded.accept();
  assert!(unrecorded.into_entries().is_empty());
}

#[test]
fn evaluation_budget_is_not_convergence() {
  let (problem, report) = minimize(
    Rosenbrock::new(),
    LmOptions {
      patience: 1,
      ..LmOptions::default()
    },
  );
  assert!(!report.converged, "{report:?}");
  assert_eq!(report.reason, TerminationReason::MaxCostEvals);
  assert_eq!(report.message(), "evaluation budget exhausted");
  assert_eq!(report.evaluations, 3);
  assert!(problem.x.iter().all(|v| v.is_finite()));
}

#[test]
fn an_iteration_cap_below_the_budget_stops_the_fit() {
  let options = |max_iterations| LmOptions {
    max_iterations,
    ..LmOptions::default()
  };
  let (free, free_report) = minimize(Rosenbrock::new(), options(None));
  let (capped, report) = minimize(Rosenbrock::new(), options(Some(2)));
  assert!(free_report.converged);
  assert!(!report.converged, "{report:?}");
  assert_eq!(report.reason, TerminationReason::MaxIter);
  assert!(report.evaluations < free_report.evaluations);
  assert!(distance_to_optimum(&capped.x) > distance_to_optimum(&free.x));
  let fit = report
    .calibrated(capped.x.to_vec(), &[1.0], &[1.0], &LossMetric::ALL, vec![])
    .unwrap();
  assert!(fit.iterations.unwrap() <= 2, "{:?}", fit.iterations);
  assert!(fit.iterations.unwrap() < report.evaluations);
}

#[test]
fn failed_trial_restores_the_last_accepted_parameters() {
  let mut problem = Rosenbrock::new();
  let initial = problem.params();
  problem.fail_at = Some(2);
  let (problem, report) = minimize(problem, LmOptions::default());
  assert!(!report.converged);
  assert_eq!(report.reason, TerminationReason::SolverFailed);
  assert_eq!(report.message(), "solver failed");
  assert_eq!(report.evaluations, 2);
  assert_eq!(problem.params(), initial);
}

#[test]
fn failed_initial_evaluation_is_reported_without_panicking() {
  let mut problem = Rosenbrock::new();
  problem.fail_at = Some(1);
  let (problem, report) = minimize(problem, LmOptions::default());
  assert!(!report.converged && !report.started);
  assert_eq!(report.evaluations, 1);
  let fit = report.calibrated(problem.x.to_vec(), &[1.0], &[1.0], &[], vec![]);
  assert!(
    matches!(fit, Err(QuantError::Optimizer { .. })),
    "a fit with no iterate is an error, got {fit:?}"
  );
}

#[test]
fn nonfinite_residual_is_not_convergence() {
  let mut problem = Rosenbrock::new();
  problem.x[0] = f64::NAN;
  let (_, report) = minimize(problem, LmOptions::default());
  assert!(!report.converged);
  assert_eq!(report.reason, TerminationReason::SolverFailed);
}

#[test]
fn nonfinite_trial_shrinks_the_step_and_recovers() {
  for pivoted_qr in [false, true] {
    let mut problem = Rosenbrock::new();
    problem.reject_outside_domain = true;
    let (problem, report) = minimize(
      problem,
      LmOptions {
        pivoted_qr,
        ..LmOptions::default()
      },
    );
    assert!(problem.nonfinite_trials.get() > 0);
    assert!(report.converged, "{report:?}");
    assert!(distance_to_optimum(&problem.x) < 1e-8);
  }
}

#[test]
fn a_pull_appends_its_rows_and_their_scaled_jacobian() {
  let objective = Objective {
    model: RefCell::new(Rosenbrock::new()),
    coordinates: RefCell::new(None),
    evaluations: Cell::new(0),
    regularization: Some(Regularization::new(vec![0.5, 2.0], vec![4.0, 9.0])),
  };
  let x = Array1::from_vec(vec![-1.2, 1.0]);
  let residuals = Residual::residual(&&objective, &x).unwrap();
  let rows = residuals.slice(ndarray::s![2..]).to_vec();
  assert_eq!(rows, vec![2.0 * (-1.2 - 0.5), 3.0 * (1.0 - 2.0)]);
  let jacobian = Jacobian::jacobian(&&objective, &x).unwrap();
  assert_eq!(jacobian.nrows(), 4);
  let pulled = jacobian.slice(ndarray::s![2.., ..]).to_owned();
  assert_eq!(pulled, ndarray::array![[2.0, 0.0], [0.0, 3.0]]);
}

#[test]
fn a_heavy_pull_holds_its_parameter_at_the_anchor() {
  let options = LmOptions {
    regularization: Some(Regularization::new(vec![0.5, 0.0], vec![1e8, 0.0])),
    ..LmOptions::default()
  };
  let (problem, report) = minimize(Rosenbrock::new(), options);
  assert!(report.converged, "{report:?}");
  assert!((problem.x[0] - 0.5).abs() < 1e-6, "{}", problem.x[0]);
  assert!((problem.x[1] - 0.25).abs() < 1e-4, "{}", problem.x[1]);
}

#[test]
fn a_zero_weight_pull_is_the_unpulled_fit_bit_for_bit() {
  let pulled = LmOptions {
    regularization: Some(Regularization::uniform(vec![0.5, 0.5], 0.0)),
    ..LmOptions::default()
  };
  let (plain, plain_report) = minimize(Rosenbrock::new(), LmOptions::default());
  let (zero, zero_report) = minimize(Rosenbrock::new(), pulled);
  let bits = |x: &Array1<f64>| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
  assert_eq!(bits(&plain.x), bits(&zero.x));
  assert_eq!(plain_report.evaluations, zero_report.evaluations);
}

fn distance_to_optimum(x: &Array1<f64>) -> f64 {
  let distance = x - &Array1::from_elem(2, 1.0);
  distance.dot(&distance).sqrt()
}
