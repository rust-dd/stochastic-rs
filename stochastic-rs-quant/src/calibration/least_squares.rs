//! Adapts stateful calibration objectives to basin's residual and Jacobian callbacks.
//!
//! Solves $\min_x \frac12\|r(x)\|^2$ with basin's trust-region LM.
//! Uses pivoted QR for the larger fits and Cholesky for the smaller pricing objectives.
//! Reference: J. J. Moré, “The Levenberg-Marquardt Algorithm: Implementation
//! and Theory” (1978), DOI: 10.1007/BFb0067700.

use std::cell::Cell;
use std::cell::RefCell;

use basin::Executor;
use basin::Jacobian;
use basin::LevenbergMarquardt;
use basin::LmDamping;
use basin::NllsState;
use basin::Residual;
use basin::Solver;
use basin::State;
use basin::StepOutcome;
use basin::TerminationReason;
use ndarray::Array1;
use ndarray::Array2;

use crate::CalibrationLossScore;
use crate::LossMetric;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationHistory;
use crate::calibration::CalibrationOptions;

/// Internal bridge shared by the quant calibrators and the AI surrogate.
/// Model coordinates and parameter projections remain owned by the calibrator.
pub trait LeastSquaresProblem {
  fn set_params(&mut self, params: &Array1<f64>);

  fn params(&self) -> Array1<f64>;

  /// `&mut self` so a problem can keep the evaluation that [`accept`](Self::accept) records.
  fn residuals(&mut self) -> Option<Array1<f64>>;

  fn jacobian(&self) -> Option<Array2<f64>>;

  /// The point of the last `residuals` call is the start or a step the optimiser accepted; a
  /// recorded trail keeps it.
  fn accept(&mut self) {}
}

#[derive(Clone, Copy, Debug)]
pub struct LmOptions {
  pub tolerance: Option<f64>,
  pub patience: usize,
  /// Use pivoted QR for fits that need its additional rank robustness.
  pub pivoted_qr: bool,
  /// Iteration cap under the evaluation budget; `None` caps at the budget itself.
  pub max_iterations: Option<usize>,
}

impl Default for LmOptions {
  fn default() -> Self {
    Self {
      tolerance: None,
      patience: 100,
      pivoted_qr: true,
      max_iterations: None,
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub struct LmReport {
  pub converged: bool,
  /// Residual evaluations, the budget's unit.
  pub evaluations: usize,
  /// Stepper iterations, one that rejects its trial included: the cap's unit.
  pub iterations: usize,
  pub reason: TerminationReason,
}

impl LmReport {
  /// Why the fit stopped, in the crate's own words: a `CalibrationResult::message` that basin's renames do not reach.
  pub fn message(&self) -> &'static str {
    match self.reason {
      TerminationReason::SolverConverged => "converged",
      TerminationReason::MaxCostEvals | TerminationReason::MaxIter => "evaluation budget exhausted",
      TerminationReason::NumericalNoProgress => "no numerical progress",
      TerminationReason::SolverFailed => "solver failed",
      _ => "stopped without convergence",
    }
  }

  /// The fit as a `Calibrated`: loss and worst error of `model` against `market`, and this stop.
  pub(crate) fn calibrated<P>(
    &self,
    params: P,
    market: &[f64],
    model: &[f64],
    metrics: &[LossMetric],
    history: Vec<CalibrationHistory<P>>,
  ) -> Calibrated<P> {
    Calibrated::repriced(params, market, model, metrics, self.converged)
      .with_iterations(self.iterations)
      .with_message(self.message())
      .with_history(history)
  }
}

/// One fit's trail: `residuals` stages the point it evaluated, and `accept` keeps it.
pub(crate) struct Trail<P> {
  entries: Vec<CalibrationHistory<P>>,
  pending: Option<CalibrationHistory<P>>,
  negated: bool,
}

impl<P: Clone> Trail<P> {
  /// The trail of a fit whose rows are market − model.
  pub(crate) fn new() -> Self {
    Self {
      entries: Vec::new(),
      pending: None,
      negated: false,
    }
  }

  /// The trail of a fit whose rows are model − market, which an entry stores negated.
  pub(crate) fn negated() -> Self {
    Self {
      negated: true,
      ..Self::new()
    }
  }

  /// `residuals` are the fit's rows at `params`; a no-op unless `options.record_history`.
  pub(crate) fn stage(
    &mut self,
    options: &CalibrationOptions,
    market: &[f64],
    model: &[f64],
    params: &P,
    residuals: &Array1<f64>,
  ) {
    if !options.record_history {
      return;
    }
    let residuals = if self.negated {
      -residuals
    } else {
      residuals.clone()
    };
    let loss = CalibrationLossScore::compute_selected(market, model, options.loss_metrics);
    let entry = CalibrationHistory::new(self.entries.len(), params.clone(), residuals, loss);
    self.pending = Some(entry);
  }

  pub(crate) fn accept(&mut self) {
    self.entries.extend(self.pending.take());
  }

  pub(crate) fn into_entries(self) -> Vec<CalibrationHistory<P>> {
    self.entries
  }
}

struct Objective<P> {
  model: RefCell<P>,
  coordinates: RefCell<Option<Array1<f64>>>,
  evaluations: Cell<usize>,
}

impl<P: LeastSquaresProblem> Objective<P> {
  fn at(&self, x: &Array1<f64>) -> std::cell::RefMut<'_, P> {
    let mut coordinates = self.coordinates.borrow_mut();
    let mut model = self.model.borrow_mut();
    if coordinates.as_ref() != Some(x) {
      model.set_params(x);
      *coordinates = Some(x.clone());
    }
    // Reusing the same coordinates preserves the surrogate's cached forward pass.
    model
  }
}

impl<P: LeastSquaresProblem> Residual for &Objective<P> {
  type Param = Array1<f64>;
  type Output = Array1<f64>;
  type Error = ();

  fn residual(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
    self.evaluations.set(self.evaluations.get() + 1);
    let residuals = self.at(x).residuals().ok_or(())?;
    // Nonfinite trial residuals reject the step and shrink basin's trust radius.
    // A nonfinite starting point has no accepted fit to recover from.
    if residuals.is_empty()
      || (self.evaluations.get() == 1 && residuals.iter().any(|v| !v.is_finite()))
    {
      return Err(());
    }
    Ok(residuals)
  }
}

impl<P: LeastSquaresProblem> Jacobian for &Objective<P> {
  type Jacobian = Array2<f64>;

  fn jacobian(&self, x: &Self::Param) -> Result<Self::Jacobian, Self::Error> {
    let jacobian = self.at(x).jacobian().ok_or(())?;
    if jacobian.iter().any(|v| !v.is_finite()) {
      return Err(());
    }
    Ok(jacobian)
  }
}

/// Runs basin with the calibrators' existing relative tolerances and residual budget.
pub fn minimize<P: LeastSquaresProblem>(problem: P, options: LmOptions) -> (P, LmReport) {
  let initial = problem.params();
  let objective = Objective {
    model: RefCell::new(problem),
    coordinates: RefCell::new(None),
    evaluations: Cell::new(0),
  };
  let tolerance = options.tolerance.unwrap_or(30.0 * f64::EPSILON);
  let gradient_tolerance = if options.tolerance.is_some() {
    0.0
  } else {
    tolerance
  };
  let budget = options
    .patience
    .saturating_mul(initial.len().saturating_add(1));
  let max_iter = options.max_iterations.map_or(budget, |n| n.min(budget));
  if options.patience == 0 || !tolerance.is_finite() || tolerance < 0.0 || initial.is_empty() {
    return (
      objective.model.into_inner(),
      LmReport {
        converged: false,
        evaluations: 0,
        iterations: 0,
        reason: TerminationReason::SolverFailed,
      },
    );
  }
  let solver = LevenbergMarquardt::<Array1<f64>, Array2<f64>>::new()
    .with_damping(LmDamping::TrustRegion)
    .with_absolute_gradient_tolerance(None)
    .with_gradient_orthogonality_tolerance(gradient_tolerance)
    .with_relative_model_reduction_tolerance(tolerance)
    .with_relative_trust_radius_tolerance(tolerance);
  let (best, reason, iterations) = if options.pivoted_qr {
    run(
      &objective,
      solver.with_pivoted_qr(),
      initial,
      budget,
      max_iter,
    )
  } else {
    run(&objective, solver, initial, budget, max_iter)
  };
  let mut model = objective.model.into_inner();
  // A rejected or failed trial may be the last callback. Return the accepted fit.
  model.set_params(&best);
  (
    model,
    LmReport {
      converged: reason == TerminationReason::SolverConverged,
      evaluations: objective.evaluations.get(),
      iterations,
      reason,
    },
  )
}

fn run<'a, P, So>(
  objective: &'a Objective<P>,
  solver: So,
  initial: Array1<f64>,
  budget: usize,
  max_iter: usize,
) -> (Array1<f64>, TerminationReason, usize)
where
  P: LeastSquaresProblem,
  So: Solver<&'a Objective<P>, NllsState<Array1<f64>>, Error = ()>,
{
  let mut best = initial.clone();
  let mut iterations = 0;
  let reason = match Executor::new(objective, solver, NllsState::new(initial))
    .max_iter(max_iter as u64)
    .max_cost_evals(budget as u64)
    .into_stepper()
  {
    Err(()) => TerminationReason::SolverFailed,
    Ok(mut stepper) => {
      // `into_stepper` evaluated the start last, so the trail opens with it.
      objective.model.borrow_mut().accept();
      loop {
        let cost = stepper.state().best_cost();
        match stepper.step() {
          Err(()) => break TerminationReason::SolverFailed,
          Ok(outcome) => {
            // basin's LM moves its iterate only to a trial that lowers the cost.
            if stepper.state().best_cost() < cost {
              best.clone_from(stepper.state().best_param());
              objective.model.borrow_mut().accept();
            }
            iterations = stepper.state().iter() as usize;
            if let StepOutcome::Stopped(reason) = outcome {
              break reason;
            }
          }
        }
      }
    }
  };
  (best, reason, iterations)
}

#[cfg(test)]
mod tests {
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
    let fit = report.calibrated(
      problem.x.to_vec(),
      &[1.0, 2.0],
      &[1.5, 2.0],
      &LossMetric::ALL,
      vec![],
    );
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
    let fit = report.calibrated(capped.x.to_vec(), &[1.0], &[1.0], &LossMetric::ALL, vec![]);
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
    let (_, report) = minimize(problem, LmOptions::default());
    assert!(!report.converged);
    assert_eq!(report.evaluations, 1);
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

  fn distance_to_optimum(x: &Array1<f64>) -> f64 {
    let distance = x - &Array1::from_elem(2, 1.0);
    distance.dot(&distance).sqrt()
  }
}
