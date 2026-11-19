use std::fmt;

use basin::Executor;
use basin::LbfgsState;
use basin::Lbfgsb;
use basin::TerminationReason;
use ndarray::Array1;
use ndarray::ArrayView1;

use self::problem::Beside;
use self::problem::MleProblem;
use self::step_back::SteppingBack;
use super::DiffusionModel;
use super::density::DensityApprox;
use crate::StatsError;
use crate::input::validate_view;
use crate::optim::more_thuente;

/// Result of maximum likelihood estimation.
#[derive(Clone, Debug)]
pub struct MleFit {
  /// Estimated parameter vector.
  pub params: Array1<f64>,
  /// Parameter names.
  pub param_names: Vec<String>,
  /// Maximised log-likelihood value.
  pub log_likelihood: f64,
  /// Sample size (number of transitions).
  pub nobs: usize,
  /// Akaike Information Criterion.
  pub aic: f64,
  /// Bayesian Information Criterion.
  pub bic: f64,
  /// No step along one parameter gains `1e-7`, or the sum's rounding on a long path: stationary
  /// along each parameter, not global. `false` leaves the best point found in `params`.
  pub converged: bool,
  /// `converged`, or why not: `not stationary`, `stopped at the bound of <name>` (the parameter
  /// a step off its bound gains along), `stopped at the edge of the law's support`, …
  pub message: String,
  /// L-BFGS-B iterations summed over every search run, kept or not, at most 200 each and 16
  /// searches. `0` when the model has no free parameters (trivially `converged = true`).
  pub iterations: usize,
}

impl fmt::Display for MleFit {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    writeln!(f, "MLE Result")?;
    writeln!(f, "----------")?;
    for (name, val) in self.param_names.iter().zip(&self.params) {
      writeln!(f, "  {:<12} = {:.6}", name, val)?;
    }
    writeln!(f, "  log-lik      = {:.4}", self.log_likelihood)?;
    writeln!(f, "  AIC          = {:.4}", self.aic)?;
    writeln!(f, "  BIC          = {:.4}", self.bic)?;
    writeln!(f, "  sample size  = {}", self.nobs)?;
    writeln!(f, "  converged    = {}", self.converged)?;
    writeln!(f, "  message      = {}", self.message)?;
    writeln!(f, "  iterations   = {}", self.iterations)?;
    Ok(())
  }
}

/// What a further search must gain in log-likelihood to replace the estimate it started from.
const GAIN: f64 = 1e-6;

/// The most a step along one parameter gains in log-likelihood beside a converged estimate: on
/// 117,768 fits the stationary ones read `2.3e-9` or less, and a stall anything up to 94.
const STATIONARY: f64 = 1e-7;

/// Roundings of the sum a gain must clear: on 366 fits of `1e4` to `3e6` points the walks beside
/// an optimum gain `0.30` of one or less, and beside the stalls they flag `1.2` or more.
const ROUNDINGS: f64 = 1.0;

/// Searches after the first: each starts where the last one left a parameter unidentified.
const STAGES: usize = 3;

/// Searches that may run again in the units of where the last one ended.
const RESCALES: usize = 12;

/// Where an L-BFGS-B search ended: the best point, the objective there and the fit signals.
struct Search {
  params: Vec<f64>,
  cost: f64,
  converged: bool,
  reason: TerminationReason,
  iterations: usize,
}

/// Why a fit stopped, `Converged` alone being `converged = true`.
enum Stop {
  Converged,
  Search(TerminationReason),
  NotStationary,
  Bound(usize),
  Edge,
  Outside,
}

impl Stop {
  /// The stop in the words of quant's calibrators, `names` the model's parameters.
  fn message(&self, names: &[&str]) -> String {
    let words = match self {
      Self::Converged => "converged",
      Self::Search(TerminationReason::MaxIter | TerminationReason::MaxCostEvals) => {
        "evaluation budget exhausted"
      }
      Self::Search(TerminationReason::NumericalNoProgress) => "no numerical progress",
      Self::Search(TerminationReason::SolverFailed) => "solver failed",
      Self::Search(_) | Self::Outside => "stopped without convergence",
      Self::NotStationary => "not stationary",
      Self::Bound(held) => return format!("stopped at the bound of {}", names[*held]),
      Self::Edge => "stopped at the edge of the law's support",
    };
    words.into()
  }
}

impl Beside {
  /// The stop the walks read, `held` saying that the parameter of the largest gain is on a bound.
  fn stop(&self, held: bool) -> Stop {
    match () {
      _ if self.gain <= STATIONARY.max(ROUNDINGS * self.rounding) => Stop::Converged,
      _ if held => Stop::Bound(self.along),
      _ if self.walled => Stop::Edge,
      _ => Stop::NotStationary,
    }
  }
}

/// The walks beside `at` and the stop they give a search that converged there: the solver's own
/// stop passes a stall, so the likelihood beside the estimate decides.
fn verdict(problem: &MleProblem<'_>, at: &[f64], lower: &[f64], upper: &[f64]) -> (Beside, Stop) {
  let beside = problem.beside(at);
  let i = beside.along;
  let held = lower[i] < upper[i] && (at[i] <= lower[i] || at[i] >= upper[i]);
  let stop = beside.stop(held);
  (beside, stop)
}

/// One L-BFGS-B search of at most 200 iterations from `start`, with an empty memory, in the
/// problem's units: there the gradient tolerance is relative to each parameter's size.
fn search(problem: MleProblem<'_>, start: &[f64]) -> Search {
  let units = problem.units();
  let solver = Lbfgsb::with_line_search(SteppingBack(more_thuente()))
    .with_absolute_projected_gradient_tolerance(f64::EPSILON.sqrt())
    .with_absolute_cost_change_tolerance(f64::EPSILON);
  let result = Executor::new(problem, solver, LbfgsState::new(units.of(start), 10))
    .max_iter(200)
    .run()
    .expect("MLE objective is infallible");
  let cost = result.best_cost();
  let best = units.params(result.best_param());
  let reason = result.reason;
  let (params, converged, iterations) = resolve_fit_outcome(best, reason, result.iter());
  Search {
    params,
    cost,
    converged,
    reason,
    iterations,
  }
}

/// Resolves a Basin L-BFGS-B outcome into the public fit signals.
fn resolve_fit_outcome(
  best_param: Vec<f64>,
  reason: TerminationReason,
  iterations: u64,
) -> (Vec<f64>, bool, usize) {
  let converged = matches!(
    reason,
    TerminationReason::CostTolerance
      | TerminationReason::ProjectedGradientTolerance
      | TerminationReason::SolverConverged
  );
  (best_param, converged, iterations as usize)
}

/// L-BFGS-B MLE of `model`, left at the estimate, in the start's units, up to 16 searches; `Err`
/// below 2 points, a non-finite one or likelihood; panics unless `bounds` has a pair per parameter.
/// $$
/// -\sum_{i=1}^{N} \ln p(X_{t_i}\mid X_{t_{i-1}};\theta,\Delta t)
/// $$
/// Reference: Liu, Nocedal (1989), Mathematical Programming 45, 503-528. DOI: 10.1007/BF01589116
pub fn fit_mle(
  model: &mut dyn DiffusionModel,
  sample: ArrayView1<f64>,
  dt: f64,
  density: DensityApprox,
  param_bounds: Option<Vec<(f64, f64)>>,
) -> Result<MleFit, StatsError> {
  let bounds = param_bounds.unwrap_or_else(|| model.param_bounds());
  let n_params = model.num_params();
  assert!(
    bounds.len() == n_params,
    "bounds must satisfy `bounds.len() == model.num_params()`, got bounds.len() = {}, model.num_params() = {n_params}",
    bounds.len()
  );
  validate_view(sample, 2)?;
  let n_transitions = sample.len() - 1;

  let x0 = model.params();

  let (best_params, stop, iterations) = if n_params == 0 {
    (x0.to_vec(), Stop::Converged, 0)
  } else {
    let init: Vec<f64> = x0.to_vec();

    let lower = bounds.iter().map(|bound| bound.0).collect::<Vec<_>>();
    let upper = bounds.iter().map(|bound| bound.1).collect::<Vec<_>>();
    let problem = MleProblem::new(&mut *model, sample, dt, density, &lower, &upper, &init);
    let mut units = problem.units();
    let mut fit = search(problem, &init);
    let mut iterations = fit.iterations;
    // A search that ends far from its units in some parameter ran in the wrong unit for it: it
    // runs again from the estimate in the estimate's units, unless that costs likelihood.
    for _ in 0..RESCALES {
      if !units.far(&fit.params) {
        break;
      }
      let from = fit.params.clone();
      let problem = MleProblem::new(&mut *model, sample, dt, density, &lower, &upper, &from);
      units = problem.units();
      let next = search(problem, &from);
      iterations += next.iterations;
      if next.cost > fit.cost {
        break;
      }
      fit = next;
    }
    // A search can stall where a parameter has no gradient, as a level at a speed near zero: it
    // runs again from the start's value of each such parameter, and the higher likelihood stays.
    for _ in 0..STAGES {
      if !fit.converged {
        break;
      }
      let problem = MleProblem::new(&mut *model, sample, dt, density, &lower, &upper, &init);
      let from = problem.loosened(&fit.params, &init);
      if from == fit.params {
        break;
      }
      let next = search(problem, &from);
      iterations += next.iterations;
      if fit.cost - next.cost <= GAIN {
        break;
      }
      fit = next;
    }
    let stop = if fit.converged {
      let at = &fit.params;
      let problem = MleProblem::new(&mut *model, sample, dt, density, &lower, &upper, at);
      verdict(&problem, at, &lower, &upper).1
    } else {
      Stop::Search(fit.reason)
    };
    (fit.params, stop, iterations)
  };

  // A convergence signal at an infeasible iterate does not certify the
  // projected point whose likelihood and information criteria we report.
  let inside = best_params
    .iter()
    .zip(&bounds)
    .all(|(&value, &(lower, upper))| (lower..=upper).contains(&value));
  let stop = match stop {
    Stop::Converged if !inside => Stop::Outside,
    stop => stop,
  };
  let clamped = best_params
    .iter()
    .enumerate()
    .map(|(i, &x)| x.clamp(bounds[i].0, bounds[i].1))
    .collect::<Vec<_>>();

  model.set_params(&clamped);

  let mut log_lik = 0.0;
  for i in 1..sample.len() {
    let t0 = (i - 1) as f64 * dt;
    log_lik += density.ln_density(model, sample[i - 1], sample[i], t0, dt);
  }
  // The search's penalty marks a point without a likelihood; an estimate there is no fit.
  if !log_lik.is_finite() {
    return Err(StatsError::Numerical {
      stage: "log-likelihood",
    });
  }

  let k = n_params as f64;
  let n = n_transitions as f64;
  let aic = 2.0 * k - 2.0 * log_lik;
  let bic = k * n.ln() - 2.0 * log_lik;

  Ok(MleFit {
    params: Array1::from_vec(clamped),
    param_names: model.param_names().into_iter().map(String::from).collect(),
    log_likelihood: log_lik,
    nobs: n_transitions,
    aic,
    bic,
    converged: matches!(stop, Stop::Converged),
    message: stop.message(&model.param_names()),
    iterations,
  })
}

#[cfg(test)]
mod far_start_tests;
mod problem;
#[cfg(test)]
mod scale_tests;
#[cfg(test)]
mod stationary_tests;
mod step_back;
#[cfg(test)]
mod tests;
