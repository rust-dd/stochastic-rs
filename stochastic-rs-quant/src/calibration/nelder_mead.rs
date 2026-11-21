//! # Nelder–Mead with a stagnation test
//!
//! $$
//! \bar f^{k+1}-\bar f^k<-\alpha\,\lVert D(f:S^k)\rVert^2,\qquad D(f:S)=V^{-T}\delta(f:S)
//! $$
//!
//! Kelley (1999), SIAM J. Optim. 10(1), 43–55, §3: a step that misses this decrease restarts the
//! simplex down its gradient, a third miss is stagnation; a stop passes O'Neill's (1971) AS 47.

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
use crate::total_order::ascending;

/// Kelley's `α`; his `nelder.m` scales it by the first simplex's `σ₊/‖D‖`, which makes the test
/// invariant to the units of the objective and of the parameters.
const SUFFICIENT_DECREASE: f64 = 1e-4;

/// Kelley's §4 takes a third failed decrease test as stagnation.
const MAX_RESTARTS: usize = 3;

/// O'Neill's `0.001` of the start simplex's steps, his factorial probe and restart edge; the
/// crate's own choice reuses it as the extent a simplex has contracted to when it may stop.
const FACTORIAL_STEP: f64 = 1e-3;

/// A stop: the vertex costs' sample standard deviation under `tolerance·(|f₀| + tolerance)`, `f₀`
/// the start's best cost (R's `nmmin`), on a simplex within `0.001` of the start's step per axis.
#[derive(Clone)]
pub(crate) struct SimplexStandardDeviation {
  tolerance: f64,
  start: Rc<Cell<Option<f64>>>,
  steps: Rc<Vec<f64>>,
}

impl SimplexStandardDeviation {
  pub(crate) fn new(tolerance: f64, steps: Vec<f64>) -> Option<Self> {
    let (start, steps) = (Rc::new(Cell::new(None)), Rc::new(steps));
    let valid = tolerance.is_finite() && tolerance >= 0.0;
    valid.then_some(Self {
      tolerance,
      start,
      steps,
    })
  }

  fn check<S>(&self, state: &S) -> Option<TerminationReason>
  where
    S: SimplexState<Float = f64, Param = Vec<f64>>,
  {
    let start = self.start.get().unwrap_or(state.costs()[0]);
    self.start.set(Some(start));
    let threshold = self.tolerance * (start.abs() + self.tolerance);
    let contracted = extents(state.vertices())
      .iter()
      .zip(self.steps.iter())
      .all(|(extent, step)| *extent <= FACTORIAL_STEP * step);
    match sample_standard_deviation(state.costs()) {
      Some(sd) => (sd < threshold && contracted).then_some(TerminationReason::SimplexTolerance),
      // Undefined spread must not silently disable the stopping rule.
      None => Some(TerminationReason::SolverFailed),
    }
  }
}

/// The largest distance of a vertex from the first along each axis.
fn extents(vertices: &[Vec<f64>]) -> Vec<f64> {
  let distance = |axis: usize| {
    vertices
      .iter()
      .map(move |v| (v[axis] - vertices[0][axis]).abs())
  };
  (0..vertices[0].len())
    .map(|axis| f64::max_or_nan_fold(distance(axis)))
    .collect()
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
  /// The spread and the simplex are small and no point `0.001` of a start step away is lower.
  Converged,
  /// No such point costs another value: the cost does not depend on its parameters there.
  Flat,
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
      Self::Flat => "flat objective",
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
      ascending(&a, &b)
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

enum Factorial {
  Minimum,
  Flat,
  Lower(Vec<f64>),
}

/// O'Neill's factorial test: the cost `0.001·step` to each side of `best` along every axis.
fn factorial_test<P>(problem: &P, best: &[f64], cost: f64, steps: &[f64]) -> Factorial
where
  P: CostFunction<Param = Vec<f64>, Output = f64, Error = Infallible>,
{
  let mut moved = false;
  for (axis, step) in steps.iter().enumerate() {
    for side in [FACTORIAL_STEP * step, -FACTORIAL_STEP * step] {
      let mut probe = best.to_vec();
      probe[axis] += side;
      let Ok(beside) = problem.cost(&probe);
      if beside < cost {
        return Factorial::Lower(probe);
      }
      moved |= beside != cost;
    }
  }
  if moved {
    Factorial::Minimum
  } else {
    Factorial::Flat
  }
}

/// The least step of a start simplex along a rate or a volatility parameter: a multiple of a
/// start on its floor would leave the cost flat along that axis.
pub(crate) const LEAST_START_STEP: f64 = 1e-4;

/// A start vertex `factor` times `x` along its axis, or `LEAST_START_STEP` past it.
pub(crate) fn start_vertex(x: f64, factor: f64) -> f64 {
  let step = x * (factor - 1.0);
  x + if step.abs() < LEAST_START_STEP {
    LEAST_START_STEP
  } else {
    step
  }
}

fn mean(values: &[f64]) -> f64 {
  values.iter().sum::<f64>() / values.len() as f64
}

/// Nelder–Mead from `simplex` under Kelley's decrease test, to a stop that passes O'Neill's
/// factorial test; each restart from a lower point counts as an iteration, so the cap bounds them.
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
  let steps = extents(&simplex);
  let Some(criterion) = SimplexStandardDeviation::new(sd_tolerance, steps.clone()) else {
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
        let best = state.best_param().clone();
        let stop = match reason {
          TerminationReason::SimplexTolerance => {
            match factorial_test(&problem, &best, state.best_cost(), &steps) {
              Factorial::Minimum => SimplexStop::Converged,
              Factorial::Flat => SimplexStop::Flat,
              Factorial::Lower(_) if iterations == max_iters => SimplexStop::IterationCap,
              Factorial::Lower(lower) => {
                iterations += 1;
                simplex = vec![lower.clone()];
                for (axis, step) in steps.iter().enumerate() {
                  let mut vertex = lower.clone();
                  vertex[axis] += FACTORIAL_STEP * step;
                  simplex.push(vertex);
                }
                continue 'run;
              }
            }
          }
          TerminationReason::MaxIter => SimplexStop::IterationCap,
          _ => SimplexStop::Undefined,
        };
        break 'run (stop, best);
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
