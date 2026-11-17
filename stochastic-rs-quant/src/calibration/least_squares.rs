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
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationHistory;
use crate::calibration::CalibrationOptions;
use crate::calibration::Regularization;

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

  /// The parameters a Tikhonov pull acts on, in the calibrator's natural order; by default the
  /// optimiser's coordinates.
  fn natural_params(&self) -> Vec<f64> {
    self.params().to_vec()
  }

  /// `∂θ/∂x` of [`natural_params`](Self::natural_params): a row per parameter, a column per
  /// coordinate; the identity by default.
  fn natural_jacobian(&self) -> Array2<f64> {
    Array2::eye(self.params().len())
  }
}

#[derive(Clone, Debug)]
pub struct LmOptions {
  pub tolerance: Option<f64>,
  pub patience: usize,
  /// Use pivoted QR for fits that need its additional rank robustness.
  pub pivoted_qr: bool,
  /// Iteration cap under the evaluation budget; `None` caps at the budget itself.
  pub max_iterations: Option<usize>,
  /// Rows `√λⱼ(θⱼ − θ⁰ⱼ)` on the problem's natural parameters, appended to its residuals.
  pub regularization: Option<Regularization>,
}

impl Default for LmOptions {
  fn default() -> Self {
    Self {
      tolerance: None,
      patience: 100,
      pivoted_qr: true,
      max_iterations: None,
      regularization: None,
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
  /// The start's residuals and Jacobian evaluated finite; without them the fit has no iterate.
  pub started: bool,
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

  /// The fit as a `Calibrated`: loss and worst error of `model` against `market`, and this stop;
  /// `Optimizer` when the start left the optimiser nothing to step from.
  pub(crate) fn calibrated<P>(
    &self,
    params: P,
    market: &[f64],
    model: &[f64],
    metrics: &[LossMetric],
    history: Vec<CalibrationHistory<P>>,
  ) -> Result<Calibrated<P>, QuantError> {
    if !self.started {
      return Err(QuantError::Optimizer {
        what: "levenberg-marquardt fit",
        reason: "the start's residuals or Jacobian are not finite".into(),
      });
    }
    Ok(
      Calibrated::repriced(params, market, model, metrics, self.converged)
        .with_iterations(self.iterations)
        .with_message(self.message())
        .with_history(history),
    )
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
  /// An active pull only, so a zero-weight one leaves the fit as it is without one.
  regularization: Option<Regularization>,
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
    let mut model = self.at(x);
    let residuals = model.residuals().filter(|r| !r.is_empty()).ok_or(())?;
    let residuals = match &self.regularization {
      Some(reg) => reg.augment_residuals(residuals, &model.natural_params()),
      None => residuals,
    };
    // Nonfinite trial residuals reject the step and shrink basin's trust radius.
    // A nonfinite starting point has no accepted fit to recover from.
    if self.evaluations.get() == 1 && residuals.iter().any(|v| !v.is_finite()) {
      return Err(());
    }
    Ok(residuals)
  }
}

impl<P: LeastSquaresProblem> Jacobian for &Objective<P> {
  type Jacobian = Array2<f64>;

  fn jacobian(&self, x: &Self::Param) -> Result<Self::Jacobian, Self::Error> {
    let model = self.at(x);
    let jacobian = model.jacobian().ok_or(())?;
    let jacobian = match &self.regularization {
      Some(reg) => reg.augment_jacobian(jacobian, reg.jacobian_rows(model.natural_jacobian())),
      None => jacobian,
    };
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
    regularization: options.regularization.filter(Regularization::is_active),
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
        started: false,
      },
    );
  }
  let solver = LevenbergMarquardt::<Array1<f64>, Array2<f64>>::new()
    .with_damping(LmDamping::TrustRegion)
    .with_absolute_gradient_tolerance(None)
    .with_gradient_orthogonality_tolerance(gradient_tolerance)
    .with_relative_model_reduction_tolerance(tolerance)
    .with_relative_trust_radius_tolerance(tolerance);
  let (best, reason, iterations, started) = if options.pivoted_qr {
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
      started,
    },
  )
}

fn run<'a, P, So>(
  objective: &'a Objective<P>,
  solver: So,
  initial: Array1<f64>,
  budget: usize,
  max_iter: usize,
) -> (Array1<f64>, TerminationReason, usize, bool)
where
  P: LeastSquaresProblem,
  So: Solver<&'a Objective<P>, NllsState<Array1<f64>>, Error = ()>,
{
  let mut best = initial.clone();
  let mut iterations = 0;
  let mut started = false;
  let reason = match Executor::new(objective, solver, NllsState::new(initial))
    .max_iter(max_iter as u64)
    .max_cost_evals(budget as u64)
    .into_stepper()
  {
    Err(()) => TerminationReason::SolverFailed,
    Ok(mut stepper) => {
      started = true;
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
  (best, reason, iterations, started)
}

#[cfg(test)]
pub(crate) mod tests;
