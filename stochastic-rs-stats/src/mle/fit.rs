use std::fmt;

use basin::Executor;
use basin::LbfgsState;
use basin::Lbfgsb;
use basin::TerminationReason;
use ndarray::Array1;
use ndarray::ArrayView1;

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
  /// Whether the L-BFGS-B run reached a recognised convergence criterion.
  ///
  /// `false` means the optimiser did not report clean convergence;
  /// [`params`](Self::params) then holds the best point found so far,
  /// which may equal the initial guess only if nothing better was ever
  /// found. If the optimiser reaches the iteration limit or the line search
  /// cannot make progress,
  /// [`params`](Self::params) still contains the best feasible point found.
  /// Inspect this field before trusting [`params`](Self::params) downstream.
  pub converged: bool,
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
    writeln!(f, "  iterations   = {}", self.iterations)?;
    Ok(())
  }
}

/// What a further search must gain in log-likelihood to replace the estimate it started from.
const GAIN: f64 = 1e-6;

/// Searches after the first: each starts where the last one left a parameter unidentified.
const STAGES: usize = 3;

/// Searches that may run again in the units of where the last one ended.
const RESCALES: usize = 12;

/// The ratio to its unit, either way, past which a parameter's estimate asks for new units.
const FAR: f64 = 8.0;

/// Where an L-BFGS-B search ended: the best point, the objective there and the fit signals.
struct Search {
  params: Vec<f64>,
  cost: f64,
  converged: bool,
  iterations: usize,
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
  let (params, converged, iterations) = resolve_fit_outcome(best, result.reason, result.iter());
  Search {
    params,
    cost,
    converged,
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

/// L-BFGS-B MLE of `model`, left at the estimate, in the start's units, up to 16 searches; check
/// `converged`. `Err` below 2 points, a non-finite one or log-likelihood; panics on a bound count.
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

  let (best_params, converged, iterations) = if n_params == 0 {
    (x0.to_vec(), true, 0)
  } else {
    let init: Vec<f64> = x0.to_vec();

    let lower = bounds.iter().map(|bound| bound.0).collect::<Vec<_>>();
    let upper = bounds.iter().map(|bound| bound.1).collect::<Vec<_>>();
    let mut fit = search(
      MleProblem::new(&mut *model, sample, dt, density, &lower, &upper, &init),
      &init,
    );
    let mut iterations = fit.iterations;
    // A search that ends far from its start in some parameter ran in the wrong unit for it: it
    // runs again from the estimate in the estimate's units, unless that costs likelihood.
    let mut units = init.clone();
    for _ in 0..RESCALES {
      let far =
        fit.params.iter().zip(&units).any(|(&p, &u)| {
          p != 0.0 && u != 0.0 && !(1.0 / FAR..=FAR).contains(&(p.abs() / u.abs()))
        });
      if !far {
        break;
      }
      units = fit.params.clone();
      let problem = MleProblem::new(&mut *model, sample, dt, density, &lower, &upper, &units);
      let next = search(problem, &units);
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
    (fit.params, fit.converged, iterations)
  };

  // A convergence signal at an infeasible iterate does not certify the
  // projected point whose likelihood and information criteria we report.
  let converged = converged
    && best_params
      .iter()
      .zip(&bounds)
      .all(|(&value, &(lower, upper))| (lower..=upper).contains(&value));
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
    converged,
    iterations,
  })
}

#[cfg(test)]
mod far_start_tests;
mod problem;
#[cfg(test)]
mod scale_tests;
mod step_back;
#[cfg(test)]
mod tests;
