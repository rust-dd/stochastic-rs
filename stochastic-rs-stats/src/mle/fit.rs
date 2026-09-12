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
use crate::optim::more_thuente;

/// Result of maximum likelihood estimation.
#[derive(Clone, Debug)]
pub struct MleResult {
  /// Estimated parameter vector.
  pub params: Array1<f64>,
  /// Parameter names.
  pub param_names: Vec<String>,
  /// Maximised log-likelihood value.
  pub log_likelihood: f64,
  /// Sample size (number of transitions).
  pub sample_size: usize,
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
  /// Number of L-BFGS-B iterations performed. `0` when the model has no free
  /// parameters to fit (trivially `converged = true` in that case).
  pub iterations: usize,
}

impl fmt::Display for MleResult {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    writeln!(f, "MLE Result")?;
    writeln!(f, "----------")?;
    for (name, val) in self.param_names.iter().zip(&self.params) {
      writeln!(f, "  {:<12} = {:.6}", name, val)?;
    }
    writeln!(f, "  log-lik      = {:.4}", self.log_likelihood)?;
    writeln!(f, "  AIC          = {:.4}", self.aic)?;
    writeln!(f, "  BIC          = {:.4}", self.bic)?;
    writeln!(f, "  sample size  = {}", self.sample_size)?;
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

impl MleProblem<'_> {
  fn clamp(&self, params: &[f64]) -> Vec<f64> {
    params
      .iter()
      .enumerate()
      .map(|(i, &value)| value.clamp(self.lower[i], self.upper[i]))
      .collect()
  }

  fn eval_nll(&self, params: &[f64]) -> f64 {
    // Line-search probes can leave the box, even with a bounded solver.
    let params = self.clamp(params);
    let mut model = self.model.lock();
    model.set_params(&params);
    let mut sum = 0.0;
    for i in 1..self.sample.len() {
      let t0 = (i - 1) as f64 * self.dt;
      let d = self
        .density
        .density(&**model, self.sample[i - 1], self.sample[i], t0, self.dt);
      sum -= d.max(1e-30).ln();
    }
    if sum.is_finite() { sum } else { 1e30 }
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

/// Resolves a Basin L-BFGS-B outcome into the public fit signals.
fn resolve_fit_outcome(
  best_param: Vec<f64>,
  reason: TerminationReason,
  iterations: u64,
) -> (Vec<f64>, bool, usize) {
  let converged = matches!(
    reason,
    TerminationReason::CostTolerance | TerminationReason::SolverConverged
  );
  (best_param, converged, iterations as usize)
}

/// Fit a 1-D SDE model by Maximum Likelihood Estimation.
///
/// The function minimises the negative log-likelihood
///
/// $$
/// -\sum_{i=1}^{N} \ln p(X_{t_i}\mid X_{t_{i-1}};\theta,\Delta t)
/// $$
///
/// using Basin's L-BFGS-B solver with a numerical gradient and box constraints.
///
/// # Arguments
/// * `model`        - the SDE model (parameters will be set to the MLE values on return)
/// * `sample`       - observed sample path (length N+1)
/// * `dt`           - sampling interval
/// * `density`      - transition density approximation method
/// * `param_bounds` - optional custom bounds (defaults to model's `param_bounds()`)
///
/// # Returns
/// An [`MleResult`] with estimated parameters, log-likelihood, AIC, BIC,
/// and the `converged` / `iterations` optimiser signals. Always check
/// `converged` before trusting `params` — a non-converged run returns the
/// best feasible point found rather than fabricating a successful fit.
///
/// # References
/// - Nocedal, J. (1980). *Mathematics of Computation*, 35(151), 773-782.
///   <https://doi.org/10.1090/S0025-5718-1980-0572855-7>
/// - Liu, D.C. & Nocedal, J. (1989). *Mathematical Programming*, 45, 503-528.
///   <https://doi.org/10.1007/BF01589116>
pub fn fit_mle(
  model: &mut dyn DiffusionModel,
  sample: ArrayView1<f64>,
  dt: f64,
  density: DensityApprox,
  param_bounds: Option<Vec<(f64, f64)>>,
) -> MleResult {
  let bounds = param_bounds.unwrap_or_else(|| model.param_bounds());
  let n_params = model.num_params();
  let n_transitions = sample.len() - 1;

  assert!(
    sample.len() >= 2,
    "sample must contain at least 2 observations"
  );
  assert_eq!(
    bounds.len(),
    n_params,
    "bounds length must match number of parameters"
  );

  let x0 = model.params();

  let (best_params, converged, iterations) = if n_params == 0 {
    (x0.to_vec(), true, 0)
  } else {
    let init: Vec<f64> = x0.to_vec();

    let lower = bounds.iter().map(|bound| bound.0).collect::<Vec<_>>();
    let upper = bounds.iter().map(|bound| bound.1).collect::<Vec<_>>();
    let problem = MleProblem {
      model: Mutex::new(&mut *model),
      sample,
      dt,
      density,
      lower,
      upper,
    };

    let solver = Lbfgsb::with_line_search(more_thuente())
      .with_absolute_projected_gradient_tolerance(f64::EPSILON.sqrt())
      .with_absolute_cost_change_tolerance(f64::EPSILON);
    let state = LbfgsState::new(init, 10);
    let result = Executor::new(problem, solver, state)
      .max_iter(200)
      .run()
      .expect("MLE objective is infallible");

    resolve_fit_outcome(result.best_param().clone(), result.reason, result.iter())
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
    let d = density.density(model, sample[i - 1], sample[i], t0, dt);
    log_lik += d.max(1e-30).ln();
  }

  let k = n_params as f64;
  let n = n_transitions as f64;
  let aic = 2.0 * k - 2.0 * log_lik;
  let bic = k * n.ln() - 2.0 * log_lik;

  MleResult {
    params: Array1::from_vec(clamped),
    param_names: model.param_names().into_iter().map(String::from).collect(),
    log_likelihood: log_lik,
    sample_size: n_transitions,
    aic,
    bic,
    converged,
    iterations,
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_stochastic::diffusion::cir::Cir;

  use super::*;

  #[test]
  fn bounded_mle_cost_rejects_negative_sigma_mirror_minimum() {
    let sample = Array1::from_vec(vec![0.1, 0.2, 0.15]);
    let mut model = Cir::new(1.0, 0.1, 3.0, 3, None, None, None, Deterministic::new(0));
    let problem = MleProblem {
      model: Mutex::new(&mut model),
      sample: sample.view(),
      dt: 0.01,
      density: DensityApprox::Euler,
      lower: vec![1e-4, 1e-6, 1e-6],
      upper: vec![50.0, 10.0, 10.0],
    };
    let positive = problem.eval_nll(&[1.0, 0.1, 3.0]);
    let negative = problem.eval_nll(&[1.0, 0.1, -3.0]);
    let boundary = problem.eval_nll(&[1.0, 0.1, 1e-6]);

    assert_eq!(negative, boundary);
    assert!(negative > positive);
  }

  #[test]
  fn bounded_mle_gradient_uses_the_feasible_boundary() {
    let sample = Array1::from_vec(vec![0.1, 0.2, 0.15]);
    let mut model = Cir::new(1.0, 0.1, 3.0, 3, None, None, None, Deterministic::new(0));
    let problem = MleProblem {
      model: Mutex::new(&mut model),
      sample: sample.view(),
      dt: 0.01,
      density: DensityApprox::Euler,
      lower: vec![1e-4, 1e-6, 1e-6],
      upper: vec![50.0, 10.0, 10.0],
    };
    let outside = problem.gradient(&vec![1.0, 0.1, 13.0]).unwrap();
    let boundary = problem.gradient(&vec![1.0, 0.1, 10.0]).unwrap();

    // At this upper bound the likelihood improves toward smaller sigma.
    assert!(boundary[2] > 0.0);
    assert_eq!(outside, boundary);
  }

  #[test]
  fn mle_result_signals_non_convergence_without_error() {
    let best = vec![0.5];
    let (params, converged, iterations) =
      resolve_fit_outcome(best.clone(), TerminationReason::MaxIter, 200);

    assert_eq!(params, best);
    assert!(!converged, "MaxIter must not be reported as converged");
    assert_eq!(iterations, 200);
  }

  #[test]
  fn mle_result_retains_best_point_on_solver_failure() {
    let best = vec![1.0, 2.0, 3.0];
    let (params, converged, iterations) =
      resolve_fit_outcome(best.clone(), TerminationReason::SolverFailed, 7);

    assert_eq!(params, best);
    assert!(!converged);
    assert_eq!(iterations, 7);
  }

  #[test]
  fn mle_result_signals_genuine_convergence() {
    let fitted = vec![1.23, 4.56];
    let (params, converged, iterations) =
      resolve_fit_outcome(fitted.clone(), TerminationReason::SolverConverged, 17);

    assert_eq!(params, fitted);
    assert!(converged);
    assert_eq!(iterations, 17);
  }
}
