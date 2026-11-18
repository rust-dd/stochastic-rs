//! # Nelder–Mead with a stagnation test
//!
//! $$
//! \bar f^{k+1}-\bar f^k<-\alpha\,\lVert D(f:S^k)\rVert^2,\qquad D(f:S)=V^{-T}\delta(f:S)
//! $$
//!
//! Kelley (1999), SIAM J. Optim. 10(1), 43–55, §3: a step that misses this decrease restarts the
//! simplex on orthogonal edges turned down its gradient, and a third miss is stagnation.

use std::cell::Cell;
use std::convert::Infallible;
use std::rc::Rc;

use basin::BasicSimplexState;
use basin::CostFunction;
use basin::Executor;
use basin::NelderMead;
use basin::SimplexState;
use basin::State;
use basin::StepOutcome;
use basin::TerminationReason;
use ndarray::Array1;
use stochastic_rs_distributions::RealExt;

use crate::CalibrationLossScore;
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationHistory;
use crate::calibration::CalibrationOptions;
use crate::calibration::Knob;

/// Kelley's `α`; his `nelder.m` scales it by the first simplex's `σ₊/‖D‖`, which makes the test
/// invariant to the units of the objective and of the parameters.
const SUFFICIENT_DECREASE: f64 = 1e-4;

/// Kelley's §4 takes a third failed decrease test as stagnation.
const MAX_RESTARTS: usize = 3;

/// A simplex fit stops when the sample standard deviation of its vertex costs falls below
/// `tolerance·(|f₀| + tolerance)`, `f₀` the start's best cost: Nash's (1990, Algorithm 19) scale.
#[derive(Clone)]
pub(crate) struct SimplexStandardDeviation {
  tolerance: f64,
  start: Rc<Cell<Option<f64>>>,
}

impl SimplexStandardDeviation {
  pub(crate) fn new(tolerance: f64) -> Option<Self> {
    let start = Rc::new(Cell::new(None));
    (tolerance.is_finite() && tolerance >= 0.0).then_some(Self { tolerance, start })
  }

  fn check<S: SimplexState<Float = f64>>(&self, state: &S) -> Option<TerminationReason> {
    let start = self.start.get().unwrap_or(state.costs()[0]);
    self.start.set(Some(start));
    let threshold = self.tolerance * (start.abs() + self.tolerance);
    match sample_standard_deviation(state.costs()) {
      Some(sd) => (sd < threshold).then_some(TerminationReason::SimplexTolerance),
      // Undefined spread must not silently disable the stopping rule.
      None => Some(TerminationReason::SolverFailed),
    }
  }
}

pub(crate) fn sample_standard_deviation(values: &[f64]) -> Option<f64> {
  if values.len() < 2 || values.iter().any(|value| !value.is_finite()) {
    return None;
  }
  // Scaling keeps finite costs from overflowing the mean or squared deviations.
  let scale = f64::max_or_nan_fold(values.iter().map(|value| value.abs()));
  if scale == 0.0 {
    return Some(0.0);
  }
  let mean = values.iter().map(|value| value / scale).sum::<f64>() / values.len() as f64;
  let squared_deviations = values
    .iter()
    .map(|value| (value / scale - mean).powi(2))
    .sum::<f64>();
  let sd = scale * (squared_deviations / (values.len() - 1) as f64).sqrt();
  sd.is_finite().then_some(sd)
}

/// Why a simplex fit stopped; only `Converged` is a convergence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SimplexStop {
  /// The spread fell below its tolerance with the decrease test passed or restarted from.
  Converged,
  IterationCap,
  /// Three steps missed Kelley's sufficient decrease.
  Stagnated,
  /// The simplex directions are linearly dependent, so it has no gradient.
  Collapsed,
  /// A vertex cost or the tolerance is not finite.
  Undefined,
}

impl SimplexStop {
  pub(crate) fn message(self) -> &'static str {
    match self {
      Self::Converged => "converged",
      Self::IterationCap => "iteration cap reached",
      Self::Stagnated => "stagnated",
      Self::Collapsed => "simplex collapsed",
      Self::Undefined => "simplex spread undefined",
    }
  }
}

pub(crate) struct SimplexRun {
  pub(crate) best: Vec<f64>,
  pub(crate) converged: bool,
  pub(crate) stop: SimplexStop,
  pub(crate) iterations: u64,
  /// The best vertex at the start and after each iteration; empty unless recorded.
  pub(crate) trail: Vec<Vec<f64>>,
}

struct Shared<'a, P>(&'a P);

impl<P> CostFunction for Shared<'_, P>
where
  P: CostFunction<Param = Vec<f64>, Output = f64, Error = Infallible>,
{
  type Param = Vec<f64>;
  type Output = f64;
  type Error = Infallible;

  fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
    self.0.cost(x)
  }
}

/// Kelley's simplex gradient `V⁻ᵀδ` of a sorted simplex, with the edge lengths `‖xⱼ − x₁‖`;
/// `None` when the edges are linearly dependent.
fn simplex_gradient(vertices: &[Vec<f64>], costs: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
  let n = vertices.len() - 1;
  let mut rows = (1..=n)
    .map(|j| {
      let edge = vertices[j]
        .iter()
        .zip(&vertices[0])
        .map(|(x, best)| x - best);
      edge.chain([costs[j] - costs[0]]).collect::<Vec<_>>()
    })
    .collect::<Vec<_>>();
  let norm = |row: &Vec<f64>| row[..n].iter().map(|v| v * v).sum::<f64>().sqrt();
  let edges = rows.iter().map(norm).collect::<Vec<_>>();
  for column in 0..n {
    let pivot = (column..n).max_by(|&a, &b| {
      let (a, b) = (rows[a][column].abs(), rows[b][column].abs());
      a.total_cmp(&b)
    })?;
    rows.swap(column, pivot);
    let lead = rows[column][column];
    if lead == 0.0 || !lead.is_finite() {
      return None;
    }
    for row in column + 1..n {
      let factor = rows[row][column] / lead;
      for k in column..=n {
        rows[row][k] -= factor * rows[column][k];
      }
    }
  }
  let mut gradient = vec![0.0; n];
  for row in (0..n).rev() {
    let tail = (row + 1..n)
      .map(|k| rows[row][k] * gradient[k])
      .sum::<f64>();
    gradient[row] = (rows[row][n] - tail) / rows[row][row];
  }
  gradient
    .iter()
    .all(|g| g.is_finite())
    .then_some((gradient, edges))
}

/// Kelley's oriented restart (3.5): the best vertex and one step of half the shortest edge along
/// each axis, against the gradient's sign as his `nelder.m` takes it (the paper prints it with).
fn oriented_restart(vertices: &[Vec<f64>], gradient: &[f64], edges: &[f64]) -> Vec<Vec<f64>> {
  let step = 0.5 * edges.iter().copied().fold(f64::INFINITY, f64::min);
  let mut simplex = vec![vertices[0].clone()];
  for (axis, slope) in gradient.iter().enumerate() {
    let mut vertex = vertices[0].clone();
    vertex[axis] += if *slope < 0.0 { step } else { -step };
    simplex.push(vertex);
  }
  simplex
}

fn mean(values: &[f64]) -> f64 {
  values.iter().sum::<f64>() / values.len() as f64
}

/// Nelder–Mead from `simplex` until the spread of its costs falls below `sd_tolerance` of the
/// start's cost, under Kelley's decrease test; panics unless the simplex has two vertices.
pub(crate) fn run_nelder_mead<P>(
  problem: P,
  simplex: Vec<Vec<f64>>,
  max_iters: u64,
  sd_tolerance: f64,
  record: bool,
) -> SimplexRun
where
  P: CostFunction<Param = Vec<f64>, Output = f64, Error = Infallible>,
{
  assert!(
    simplex.len() >= 2,
    "simplex must contain at least two vertices"
  );
  let mut trail = Vec::new();
  let Some(criterion) = SimplexStandardDeviation::new(sd_tolerance) else {
    let best = simplex[0].clone();
    trail.extend(record.then(|| best.clone()));
    return SimplexRun {
      best,
      converged: false,
      stop: SimplexStop::Undefined,
      iterations: 0,
      trail,
    };
  };
  let mut simplex = simplex;
  let (mut iterations, mut restarts) = (0, 0);
  let mut alpha = None;
  let (stop, best) = 'run: loop {
    let state = BasicSimplexState::from_simplex(simplex);
    let mut stepper = Executor::new(Shared(&problem), NelderMead::new(), state)
      .max_iter(max_iters - iterations)
      .stop_when({
        let criterion = criterion.clone();
        move |state| criterion.check(state)
      })
      .into_stepper()
      .expect("calibration objective is infallible");
    loop {
      if record {
        trail.push(stepper.state().best_param().clone());
      }
      let vertices = stepper.state().vertices().to_vec();
      let costs = stepper.state().costs().to_vec();
      let outcome = stepper.step().expect("calibration objective is infallible");
      let state = stepper.state();
      if let StepOutcome::Stopped(reason) = outcome {
        let stop = match reason {
          TerminationReason::SimplexTolerance => SimplexStop::Converged,
          TerminationReason::MaxIter => SimplexStop::IterationCap,
          _ => SimplexStop::Undefined,
        };
        break 'run (stop, state.best_param().clone());
      }
      iterations += 1;
      let Some((gradient, edges)) = simplex_gradient(&vertices, &costs) else {
        trail.extend(record.then(|| state.best_param().clone()));
        break 'run (SimplexStop::Collapsed, state.best_param().clone());
      };
      let squared_norm = gradient.iter().map(|g| g * g).sum::<f64>();
      let alpha = *alpha.get_or_insert_with(|| {
        let longest = f64::max_or_nan_fold(edges.iter().copied());
        match squared_norm > 0.0 {
          true => SUFFICIENT_DECREASE * longest / squared_norm.sqrt(),
          false => 0.0,
        }
      });
      // A shrink moves every vertex but the best, and Kelley's test reads the other steps.
      let replaced = state.vertices().iter().filter(|v| !vertices.contains(v));
      let decrease = mean(state.costs()) - mean(&costs);
      if replaced.count() == 1 && decrease > -alpha * squared_norm / gradient.len() as f64 {
        restarts += 1;
        if restarts == MAX_RESTARTS {
          trail.extend(record.then(|| state.best_param().clone()));
          break 'run (SimplexStop::Stagnated, state.best_param().clone());
        }
        simplex = oriented_restart(&vertices, &gradient, &edges);
        continue 'run;
      }
    }
  };
  SimplexRun {
    best,
    converged: stop == SimplexStop::Converged,
    stop,
    iterations,
    trail,
  }
}

/// A Nelder–Mead fit as a [`Calibrated`]; `price` maps a vertex to its parameters and its model
/// and market prices, for the best vertex and, when history is recorded, each one in the trail.
pub(crate) fn nelder_mead_fit<C, P>(
  cost: C,
  simplex: Vec<Vec<f64>>,
  options: &CalibrationOptions,
  default_max_iterations: usize,
  price: impl Fn(&[f64]) -> (P, Vec<f64>, Vec<f64>),
) -> Result<Calibrated<P>, QuantError>
where
  C: CostFunction<Param = Vec<f64>, Output = f64, Error = Infallible>,
{
  options.check(&[Knob::Patience])?;
  let run = run_nelder_mead(
    cost,
    simplex,
    options.max_iterations.unwrap_or(default_max_iterations) as u64,
    options.tolerance.unwrap_or(1e-10),
    options.record_history,
  );
  let history = run
    .trail
    .iter()
    .enumerate()
    .map(|(iteration, vertex)| {
      let (params, market, model) = price(vertex);
      let residuals = Array1::from_iter(market.iter().zip(&model).map(|(q, m)| q - m));
      let loss = CalibrationLossScore::compute_selected(&market, &model, options.loss_metrics);
      CalibrationHistory::new(iteration, params, residuals, loss)
    })
    .collect();
  let (params, market, model) = price(&run.best);
  let fit = Calibrated::repriced(params, &market, &model, options.loss_metrics, run.converged);
  Ok(
    fit
      .with_iterations(run.iterations as usize)
      .with_message(run.stop.message())
      .with_history(history),
  )
}

#[cfg(test)]
mod tests;
