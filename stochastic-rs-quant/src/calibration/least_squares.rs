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
use nalgebra::DMatrix;
use nalgebra::DVector;

/// Internal bridge shared by the quant calibrators and the AI surrogate.
/// Model coordinates and parameter projections remain owned by the calibrator.
pub trait LeastSquaresProblem {
  fn set_params(&mut self, params: &DVector<f64>);

  fn params(&self) -> DVector<f64>;

  fn residuals(&self) -> Option<DVector<f64>>;

  fn jacobian(&self) -> Option<DMatrix<f64>>;
}

#[derive(Clone, Copy, Debug)]
pub struct LmOptions {
  pub tolerance: Option<f64>,
  pub patience: usize,
  /// Use pivoted QR for fits that need its additional rank robustness.
  pub pivoted_qr: bool,
}

impl Default for LmOptions {
  fn default() -> Self {
    Self {
      tolerance: None,
      patience: 100,
      pivoted_qr: true,
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub struct LmReport {
  pub converged: bool,
  pub evaluations: usize,
  pub reason: TerminationReason,
}

struct Objective<P> {
  model: RefCell<P>,
  coordinates: RefCell<Option<DVector<f64>>>,
  evaluations: Cell<usize>,
}

impl<P: LeastSquaresProblem> Objective<P> {
  fn at(&self, x: &DVector<f64>) -> std::cell::Ref<'_, P> {
    let mut coordinates = self.coordinates.borrow_mut();
    if coordinates.as_ref() != Some(x) {
      self.model.borrow_mut().set_params(x);
      *coordinates = Some(x.clone());
    }
    // Reusing the same coordinates preserves the surrogate's cached forward pass.
    self.model.borrow()
  }
}

impl<P: LeastSquaresProblem> Residual for &Objective<P> {
  type Param = DVector<f64>;
  type Output = DVector<f64>;
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
  type Jacobian = DMatrix<f64>;

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
  if options.patience == 0 || !tolerance.is_finite() || tolerance < 0.0 || initial.is_empty() {
    return (
      objective.model.into_inner(),
      LmReport {
        converged: false,
        evaluations: 0,
        reason: TerminationReason::SolverFailed,
      },
    );
  }
  let solver = LevenbergMarquardt::<DVector<f64>, DMatrix<f64>>::new()
    .with_damping(LmDamping::TrustRegion)
    .with_absolute_gradient_tolerance(None)
    .with_gradient_orthogonality_tolerance(gradient_tolerance)
    .with_relative_model_reduction_tolerance(tolerance)
    .with_relative_trust_radius_tolerance(tolerance);
  let (best, reason) = if options.pivoted_qr {
    run(&objective, solver.with_pivoted_qr(), initial, budget)
  } else {
    run(&objective, solver, initial, budget)
  };
  let mut model = objective.model.into_inner();
  // A rejected or failed trial may be the last callback. Return the accepted fit.
  model.set_params(&best);
  (
    model,
    LmReport {
      converged: reason == TerminationReason::SolverConverged,
      evaluations: objective.evaluations.get(),
      reason,
    },
  )
}

fn run<'a, P, So>(
  objective: &'a Objective<P>,
  solver: So,
  initial: DVector<f64>,
  budget: usize,
) -> (DVector<f64>, TerminationReason)
where
  P: LeastSquaresProblem,
  So: Solver<&'a Objective<P>, NllsState<DVector<f64>>, Error = ()>,
{
  let mut best = initial.clone();
  let reason = match Executor::new(objective, solver, NllsState::new(initial))
    .max_iter(budget as u64)
    .max_cost_evals(budget as u64)
    .into_stepper()
  {
    Err(()) => TerminationReason::SolverFailed,
    Ok(mut stepper) => loop {
      match stepper.step() {
        Err(()) => break TerminationReason::SolverFailed,
        Ok(outcome) => {
          best.clone_from(stepper.state().best_param());
          if let StepOutcome::Stopped(reason) = outcome {
            break reason;
          }
        }
      }
    },
  };
  (best, reason)
}

#[cfg(test)]
mod tests {
  use super::*;

  struct Rosenbrock {
    x: DVector<f64>,
    calls: Cell<usize>,
    fail_at: Option<usize>,
    reject_outside_domain: bool,
    nonfinite_trials: Cell<usize>,
  }

  impl Rosenbrock {
    fn new() -> Self {
      Self {
        x: DVector::from_vec(vec![-1.2, 1.0]),
        calls: Cell::new(0),
        fail_at: None,
        reject_outside_domain: false,
        nonfinite_trials: Cell::new(0),
      }
    }
  }

  impl LeastSquaresProblem for Rosenbrock {
    fn set_params(&mut self, x: &DVector<f64>) {
      self.x.clone_from(x);
    }

    fn params(&self) -> DVector<f64> {
      self.x.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
      self.calls.set(self.calls.get() + 1);
      if self.fail_at == Some(self.calls.get()) {
        return None;
      }
      if self.reject_outside_domain && self.x[1] < -2.0 {
        self.nonfinite_trials.set(self.nonfinite_trials.get() + 1);
        return Some(DVector::from_element(2, f64::NAN));
      }
      Some(DVector::from_vec(vec![
        10.0 * (self.x[1] - self.x[0].powi(2)),
        1.0 - self.x[0],
      ]))
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
      Some(DMatrix::from_row_slice(
        2,
        2,
        &[-20.0 * self.x[0], 10.0, -1.0, 0.0],
      ))
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
      assert!((problem.x - DVector::from_element(2, 1.0)).norm() < 1e-8);
      assert_eq!(report.evaluations, problem.calls.get());
    }
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
    assert_eq!(report.evaluations, 3);
    assert!(problem.x.iter().all(|v| v.is_finite()));
  }

  #[test]
  fn failed_trial_restores_the_last_accepted_parameters() {
    let mut problem = Rosenbrock::new();
    let initial = problem.params();
    problem.fail_at = Some(2);
    let (problem, report) = minimize(problem, LmOptions::default());
    assert!(!report.converged);
    assert_eq!(report.reason, TerminationReason::SolverFailed);
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
      assert!((problem.x - DVector::from_element(2, 1.0)).norm() < 1e-8);
    }
  }
}
