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
use ndarray::ArrayView1;
use stochastic_rs_distributions::RealExt;

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
  fn set_params(&mut self, params: ArrayView1<f64>);

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

impl LmOptions {
  /// The stepper iterations a fit of `dimension` coordinates may take: `max_iterations` under
  /// the evaluation budget.
  pub(crate) fn iteration_cap(&self, dimension: usize) -> usize {
    let budget = self.patience.saturating_mul(dimension.saturating_add(1));
    self.max_iterations.map_or(budget, |n| n.min(budget))
  }
}

#[derive(Clone, Copy, Debug)]
pub struct LmReport {
  /// The solver stopped at a stationary point, or at a fit [`settled`](Self::settled) as exact;
  /// `message()` is `converged` exactly then.
  pub converged: bool,
  /// Residual evaluations, the budget's unit.
  pub evaluations: usize,
  /// Stepper iterations, one that rejects its trial included: the cap's unit.
  pub iterations: usize,
  reason: TerminationReason,
  /// The start's residuals and Jacobian evaluated finite; without them the fit has no iterate.
  pub started: bool,
  /// The fit stopped where `‖Dx‖ ≤ ε‖r‖`: no step the size of `x` moves a residual's last bit.
  pub flat: bool,
  /// MINPACK's orthogonality measure `maxⱼ |Jⱼᵀr| / (‖Jⱼ‖‖r‖)` over the columns a step lowers
  /// the cost along, where the solver stopped; NaN where it could not be evaluated, 0 otherwise.
  pub stationarity: f64,
}

/// The bound on the measure of a converged fit: the crate's own fits read 6.6e-4 or less at a
/// minimum, 1.7e-2 resting on a clamp, which stays converged, and 7.2e-2 or more short of one.
const STATIONARY: f64 = 3e-3;

/// The shares of a column's Gauss–Newton step the first-order check tries: the whole step
/// overshoots where a bound's map bends the residual.
const STEP_LADDER: [f64; 4] = [1.0, 0.25, 0.0625, 0.015625];

/// The pricers' relative accuracy, the Fourier integrals' own tolerance: an rmse this share of
/// the quotes' root mean square is noise, and its angle to the Jacobian means nothing.
const PRICER_ACCURACY: f64 = 1e-8;

/// `Jⱼᵀr / (‖Jⱼ‖‖r‖)` per column, signed; 0 for a zero column or a zero residual.
fn column_cosines(residuals: &Array1<f64>, jacobian: &Array2<f64>) -> Vec<f64> {
  let norm = residuals.dot(residuals).sqrt();
  let cosine = |column: ndarray::ArrayView1<f64>| {
    let scale = column.dot(&column).sqrt() * norm;
    if scale == 0.0 {
      0.0
    } else {
      column.dot(residuals) / scale
    }
  };
  jacobian.columns().into_iter().map(cosine).collect()
}

fn pulled_residuals<P: LeastSquaresProblem>(
  problem: &mut P,
  pull: Option<&Regularization>,
) -> Option<Array1<f64>> {
  let residuals = problem.residuals().filter(|r| !r.is_empty())?;
  Some(match pull {
    Some(reg) => reg.augment_residuals(residuals, &problem.natural_params()),
    None => residuals,
  })
}

fn pulled_jacobian<P: LeastSquaresProblem>(
  problem: &P,
  pull: Option<&Regularization>,
) -> Option<Array2<f64>> {
  let jacobian = problem.jacobian()?;
  Some(match pull {
    Some(reg) => reg.augment_jacobian(jacobian, reg.jacobian_rows(problem.natural_jacobian())),
    None => jacobian,
  })
}

/// The signed cosines of the first-order check at the problem's point, an active pull's rows
/// included as `minimize` appends them.
pub(crate) fn cosines<P: LeastSquaresProblem>(
  problem: &mut P,
  pull: Option<&Regularization>,
) -> Option<Vec<f64>> {
  let pull = pull.filter(|reg| reg.is_active());
  let residuals = pulled_residuals(problem, pull)?;
  Some(column_cosines(&residuals, &pulled_jacobian(problem, pull)?))
}

/// [`cosines`] from `jacobian`, `∂r/∂θ` in the natural parameters: a coordinate whose map has
/// rounded onto its bound keeps a column there, where the optimiser's coordinates zero it.
pub(crate) fn natural_cosines<P: LeastSquaresProblem>(
  problem: &mut P,
  jacobian: Array2<f64>,
  pull: Option<&Regularization>,
) -> Option<Vec<f64>> {
  let pull = pull.filter(|reg| reg.is_active());
  let residuals = pulled_residuals(problem, pull)?;
  let jacobian = match pull {
    Some(reg) => {
      let rows = reg.jacobian_rows(Array2::eye(jacobian.ncols()));
      reg.augment_jacobian(jacobian, rows)
    }
    None => jacobian,
  };
  Some(column_cosines(&residuals, &jacobian))
}

impl LmReport {
  /// Why the fit stopped, in the crate's own words: a `CalibrationResult::message` that basin's renames do not reach.
  pub fn message(&self) -> &'static str {
    match self.reason {
      _ if self.flat => "flat objective",
      TerminationReason::SolverConverged if self.converged => "converged",
      TerminationReason::SolverConverged => "not stationary",
      TerminationReason::MaxCostEvals | TerminationReason::MaxIter => "evaluation budget exhausted",
      TerminationReason::NumericalNoProgress => "no numerical progress",
      TerminationReason::SolverFailed => "solver failed",
      _ => "stopped without convergence",
    }
  }

  /// Whether the solver itself stopped on its tolerances, before the first-order check.
  pub(crate) fn solver_converged(&self) -> bool {
    self.reason == TerminationReason::SolverConverged
  }

  /// Where the solver stopped, a fit whose `rmse` is within `accuracy` of `market`'s root mean
  /// square is converged: the pricers' accuracy is `1e-8`, an f32 surrogate's `8·f32::EPSILON`.
  pub fn settled(self, rmse: f64, market: &[f64], accuracy: f64) -> Self {
    let size = (market.iter().map(|q| q * q).sum::<f64>() / market.len() as f64).sqrt();
    let exact = self.solver_converged() && !self.flat && rmse <= accuracy * size;
    Self {
      converged: self.converged || exact,
      ..self
    }
  }

  /// The fit as a `Calibrated`: loss and worst error of `model` against `market`, and this stop,
  /// converged only if stationary or exact; `Optimizer` when the start left nothing to step from.
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
    Ok(self.fit(params, market, model, metrics, history))
  }

  /// The fit as [`calibrated`](Self::calibrated) builds it, a start it could not evaluate included:
  /// that one reads `solver failed` at the start's parameters.
  pub(crate) fn fit<P>(
    &self,
    params: P,
    market: &[f64],
    model: &[f64],
    metrics: &[LossMetric],
    history: Vec<CalibrationHistory<P>>,
  ) -> Calibrated<P> {
    let fit = Calibrated::repriced(params, market, model, metrics, self.converged);
    let settled = self.settled(fit.loss.get(LossMetric::Rmse), market, PRICER_ACCURACY);
    let fit = Calibrated {
      converged: settled.converged,
      ..fit
    };
    fit
      .with_iterations(self.iterations)
      .with_message(settled.message())
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
    residuals: ArrayView1<f64>,
  ) {
    if !options.record_history {
      return;
    }
    let residuals = if self.negated {
      -&residuals
    } else {
      residuals.to_owned()
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
  /// `‖r‖` of the last residual call, which precedes each Jacobian basin asks for at its point.
  norm: Cell<f64>,
  /// The largest `|xⱼ|` a Jacobian was asked at, the size the flat rule gives each column.
  sizes: RefCell<Array1<f64>>,
  flat: Cell<bool>,
  /// An active pull only, so a zero-weight one leaves the fit as it is without one.
  regularization: Option<Regularization>,
}

impl<P: LeastSquaresProblem> Objective<P> {
  fn at(&self, x: &Array1<f64>) -> std::cell::RefMut<'_, P> {
    let mut coordinates = self.coordinates.borrow_mut();
    let mut model = self.model.borrow_mut();
    if coordinates.as_ref() != Some(x) {
      model.set_params(x.view());
      *coordinates = Some(x.clone());
    }
    // Reusing the same coordinates preserves the surrogate's cached forward pass.
    model
  }
}

impl<P: LeastSquaresProblem> Objective<P> {
  /// The largest `|Jⱼᵀr| / (‖Jⱼ‖‖r‖)` at `x` over the columns with a share `t` of whose
  /// Gauss–Newton step, to either side, `‖r‖²` falls by half the `t(2 − t)·cos²·‖r‖²` it predicts.
  fn stationarity(&self, x: &Array1<f64>) -> f64 {
    let rows = Residual::residual(&self, x).and_then(|r| Ok((r, Jacobian::jacobian(&self, x)?)));
    let Ok((r, j)) = rows else {
      return if self.flat.get() { 0.0 } else { f64::NAN };
    };
    let cost = r.dot(&r);
    let lowers = |column: usize, step: f64, predicted: f64| {
      let mut beside = x.clone();
      beside[column] += step;
      Residual::residual(&self, &beside).is_ok_and(|q| q.dot(&q) < cost - 0.5 * predicted)
    };
    let cosines = column_cosines(&r, &j).into_iter().enumerate();
    f64::max_or_nan_fold(cosines.map(|(column, cosine)| {
      // A bound or a kink leaves an angle no step can use; a wrong Jacobian leaves one that the
      // step against its own sign can.
      let step = cosine * cost.sqrt() / j.column(column).dot(&j.column(column)).sqrt();
      let open = cosine.abs() > STATIONARY
        && STEP_LADDER.iter().any(|share| {
          let predicted = share * (2.0 - share) * cosine * cosine * cost;
          lowers(column, -share * step, predicted) || lowers(column, share * step, predicted)
        });
      if open || cosine.is_nan() {
        cosine.abs()
      } else {
        0.0
      }
    }))
  }
}

impl<P: LeastSquaresProblem> Residual for &Objective<P> {
  type Param = Array1<f64>;
  type Output = Array1<f64>;
  type Error = ();

  fn residual(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
    self.evaluations.set(self.evaluations.get() + 1);
    let mut model = self.at(x);
    let residuals = pulled_residuals(&mut *model, self.regularization.as_ref()).ok_or(())?;
    // Nonfinite trial residuals reject the step and shrink basin's trust radius.
    // A nonfinite starting point has no accepted fit to recover from.
    if self.evaluations.get() == 1 && residuals.iter().any(|v| !v.is_finite()) {
      return Err(());
    }
    self.norm.set(residuals.dot(&residuals).sqrt());
    Ok(residuals)
  }
}

impl<P: LeastSquaresProblem> Jacobian for &Objective<P> {
  type Jacobian = Array2<f64>;

  fn jacobian(&self, x: &Self::Param) -> Result<Self::Jacobian, Self::Error> {
    let model = self.at(x);
    let jacobian = pulled_jacobian(&*model, self.regularization.as_ref()).ok_or(())?;
    if jacobian.iter().any(|v| !v.is_finite()) {
      return Err(());
    }
    // Moré's ‖Dx‖ of (8.3), D the column norms, x at its largest so far: a minimum near the
    // origin is not flat. A fit that never left the origin is sized by ‖D‖ alone.
    let mut sizes = self.sizes.borrow_mut();
    sizes.zip_mut_with(x, |size, x| *size = size.max(x.abs()));
    let origin = sizes.iter().all(|v| *v == 0.0);
    let scaled = jacobian
      .columns()
      .into_iter()
      .zip(sizes.iter())
      .map(|(column, x)| {
        let size = if origin { 1.0 } else { *x };
        column.dot(&column) * size * size
      });
    let norm = self.norm.get();
    if norm > 0.0 && scaled.sum::<f64>().sqrt() <= f64::EPSILON * norm {
      self.flat.set(true);
      return Err(());
    }
    Ok(jacobian)
  }
}

/// Runs basin with the calibrators' existing relative tolerances and residual budget.
pub fn minimize<P: LeastSquaresProblem>(problem: P, options: LmOptions) -> (P, LmReport) {
  let initial = problem.params();
  let max_iter = options.iteration_cap(initial.len());
  let objective = Objective {
    model: RefCell::new(problem),
    coordinates: RefCell::new(None),
    evaluations: Cell::new(0),
    norm: Cell::new(0.0),
    sizes: RefCell::new(Array1::zeros(initial.len())),
    flat: Cell::new(false),
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
  if options.patience == 0 || !tolerance.is_finite() || tolerance < 0.0 || initial.is_empty() {
    return (
      objective.model.into_inner(),
      LmReport {
        converged: false,
        evaluations: 0,
        iterations: 0,
        reason: TerminationReason::SolverFailed,
        started: false,
        flat: false,
        stationarity: 0.0,
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
  let stationarity = match reason == TerminationReason::SolverConverged {
    true => objective.stationarity(&best),
    false => 0.0,
  };
  let flat = objective.flat.get();
  let converged =
    reason == TerminationReason::SolverConverged && !flat && stationarity <= STATIONARY;
  let evaluations = objective.evaluations.get();
  let mut model = objective.model.into_inner();
  if flat && !started {
    model.accept();
  }
  // A rejected or failed trial may be the last callback. Return the accepted fit.
  model.set_params(best.view());
  (
    model,
    LmReport {
      converged,
      evaluations,
      iterations,
      reason,
      started: started || flat,
      flat,
      stationarity,
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
pub(crate) mod checks;

#[cfg(test)]
mod tests;
