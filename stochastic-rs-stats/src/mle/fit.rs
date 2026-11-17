use std::convert::Infallible;
use std::fmt;

use basin::BoxConstraints;
use basin::CostFunction;
use basin::Executor;
use basin::Gradient;
use basin::LbfgsState;
use basin::Lbfgsb;
use basin::TerminationReason;
use ndarray::Array1;
use ndarray::ArrayView1;
use parking_lot::Mutex;

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
  /// L-BFGS-B iterations summed over every search run, at most 200 each: the first and up to three
  /// more, kept or not. `0` when the model has no free parameters (trivially `converged = true`).
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

/// Basin problem wrapper for MLE optimisation.
struct MleProblem<'a> {
  model: Mutex<&'a mut dyn DiffusionModel>,
  sample: ArrayView1<'a, f64>,
  dt: f64,
  density: DensityApprox,
  lower: Vec<f64>,
  upper: Vec<f64>,
}

impl<'a> MleProblem<'a> {
  fn new(
    model: &'a mut dyn DiffusionModel,
    sample: ArrayView1<'a, f64>,
    dt: f64,
    density: DensityApprox,
    lower: &[f64],
    upper: &[f64],
  ) -> Self {
    Self {
      model: Mutex::new(model),
      sample,
      dt,
      density,
      lower: lower.to_vec(),
      upper: upper.to_vec(),
    }
  }

  fn clamp(&self, params: &[f64]) -> Vec<f64> {
    params
      .iter()
      .enumerate()
      .map(|(i, &value)| value.clamp(self.lower[i], self.upper[i]))
      .collect()
  }

  /// `at` with each parameter the likelihood does not hold inside its box back at `start`'s
  /// value: one whose move alone to an end of the box costs less than [`IDENTIFIED`].
  fn loosened(&self, at: &[f64], start: &[f64]) -> Vec<f64> {
    let here = self.eval_nll(at);
    let mut from = at.to_vec();
    if here >= NO_LIKELIHOOD {
      return from;
    }
    for i in 0..at.len() {
      let ends = [self.lower[i], self.upper[i]];
      let loose = ends[0] < ends[1]
        && ends.into_iter().any(|end| {
          let mut moved = at.to_vec();
          moved[i] = end;
          self.eval_nll(&moved) - here < IDENTIFIED
        });
      if loose {
        from[i] = start[i];
      }
    }
    from
  }

  fn eval_nll(&self, params: &[f64]) -> f64 {
    // Line-search probes can leave the box, even with a bounded solver.
    let params = self.clamp(params);
    let mut model = self.model.lock();
    model.set_params(&params);
    let mut sum = 0.0;
    for i in 1..self.sample.len() {
      let t0 = (i - 1) as f64 * self.dt;
      let ln_d = self
        .density
        .ln_density(&**model, self.sample[i - 1], self.sample[i], t0, self.dt);
      // A density of zero, an infinite one and none at all are no likelihood.
      if !ln_d.is_finite() {
        return NO_LIKELIHOOD;
      }
      sum -= ln_d;
    }
    if sum.is_finite() {
      ranked(sum)
    } else {
      NO_LIKELIHOOD
    }
  }
}

/// Half of `χ²₁(0.95)`, the drop in log-likelihood at the end of a 95 % likelihood interval.
const IDENTIFIED: f64 = 1.92;

/// What a further search must gain in log-likelihood to replace the estimate it started from.
const GAIN: f64 = 1e-6;

/// Searches after the first: each starts where the last one left a parameter unidentified.
const STAGES: usize = 3;

/// The objective at a point without a likelihood, above every sum [`ranked`] returns.
const NO_LIKELIHOOD: f64 = 1e100;

/// `nll` up to `1e97`, far past any realistic path's sum, and its logarithmic continuation past it,
/// at most `4.9e99`: increasing and C¹, so a finite sum ranks below the penalty and none is `∞`.
fn ranked(nll: f64) -> f64 {
  const KNEE: f64 = 1e97;
  if nll <= KNEE {
    nll
  } else {
    KNEE * (1.0 + (nll / KNEE).ln())
  }
}

impl CostFunction for MleProblem<'_> {
  type Param = Vec<f64>;
  type Output = f64;
  type Error = Infallible;

  fn cost(&self, params: &Self::Param) -> Result<Self::Output, Self::Error> {
    Ok(self.eval_nll(params))
  }
}

impl Gradient for MleProblem<'_> {
  type Gradient = Vec<f64>;

  fn gradient(&self, params: &Self::Param) -> Result<Self::Gradient, Self::Error> {
    let params = self.clamp(params);
    let n = params.len();
    let mut grad = vec![0.0; n];
    for i in 0..n {
      let h = 1e-7 * (1.0 + params[i].abs());
      let mut p_plus = params.clone();
      let mut p_minus = params.clone();
      p_plus[i] = (params[i] + h).min(self.upper[i]);
      p_minus[i] = (params[i] - h).max(self.lower[i]);
      let actual_2h = p_plus[i] - p_minus[i];
      if actual_2h > 0.0 {
        let fp = self.eval_nll(&p_plus);
        let fm = self.eval_nll(&p_minus);
        grad[i] = (fp - fm) / actual_2h;
      }
    }
    Ok(grad)
  }
}

impl BoxConstraints for MleProblem<'_> {
  fn lower(&self) -> &Self::Param {
    &self.lower
  }

  fn upper(&self) -> &Self::Param {
    &self.upper
  }
}

/// Where an L-BFGS-B search ended: the best point, the objective there and the fit signals.
struct Search {
  params: Vec<f64>,
  cost: f64,
  converged: bool,
  iterations: usize,
}

/// One L-BFGS-B search of at most 200 iterations from `start`, with an empty memory.
fn search(problem: MleProblem<'_>, start: Vec<f64>) -> Search {
  let solver = Lbfgsb::with_line_search(more_thuente())
    .with_absolute_projected_gradient_tolerance(f64::EPSILON.sqrt())
    .with_absolute_cost_change_tolerance(f64::EPSILON);
  let result = Executor::new(problem, solver, LbfgsState::new(start, 10))
    .max_iter(200)
    .run()
    .expect("MLE objective is infallible");
  let cost = result.best_cost();
  let (params, converged, iterations) =
    resolve_fit_outcome(result.best_param().clone(), result.reason, result.iter());
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

/// L-BFGS-B MLE of `model`, left at the estimate, on a path every `dt`, in up to 4 searches. `Err`
/// below 2 points, on a non-finite one or log-likelihood; panics unless one bound per parameter.
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
      MleProblem::new(&mut *model, sample, dt, density, &lower, &upper),
      init.clone(),
    );
    // A search can stall where a parameter has no gradient, as a level at a speed near zero: it
    // runs again from the start's value of each such parameter, and the higher likelihood stays.
    let mut iterations = fit.iterations;
    for _ in 0..STAGES {
      if !fit.converged {
        break;
      }
      let problem = MleProblem::new(&mut *model, sample, dt, density, &lower, &upper);
      let from = problem.loosened(&fit.params, &init);
      if from == fit.params {
        break;
      }
      let next = search(problem, from);
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
#[cfg(test)]
mod tests;
