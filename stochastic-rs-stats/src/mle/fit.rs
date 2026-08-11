use std::fmt;

use argmin::core::CostFunction;
use argmin::core::Executor;
use argmin::core::Gradient;
use argmin::core::State;
use argmin::core::TerminationReason;
use argmin::core::TerminationStatus;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ndarray::Array1;
use ndarray::ArrayView1;
use parking_lot::Mutex;

use super::DiffusionModel;
use super::density::DensityApprox;

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
  /// Whether the L-BFGS run reached a recognised convergence criterion
  /// (`SolverConverged` / `TargetCostReached`).
  ///
  /// `false` means [`params`](Self::params) is the untouched initial
  /// guess passed in via `model`'s parameters at call time: either the
  /// optimiser itself errored before completing a step, or it terminated
  /// (max iterations, an internal line-search exit, timeout, interrupt)
  /// without ever improving on that starting point. Previously this case
  /// was indistinguishable from a genuine fit — the initial guess was
  /// returned silently. Inspect this field (or propagate it) before
  /// trusting [`params`](Self::params) downstream.
  pub converged: bool,
  /// Number of L-BFGS iterations performed. `0` when the optimiser
  /// errored before its first step, or when the model has no free
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

/// Argmin problem wrapper for MLE optimisation.
struct MleProblem<'a> {
  model: Mutex<&'a mut dyn DiffusionModel>,
  sample: ArrayView1<'a, f64>,
  dt: f64,
  density: DensityApprox,
  bounds: Vec<(f64, f64)>,
}

impl MleProblem<'_> {
  fn clamp(&self, params: &[f64]) -> Vec<f64> {
    params
      .iter()
      .enumerate()
      .map(|(i, &x)| x.clamp(self.bounds[i].0, self.bounds[i].1))
      .collect()
  }

  fn eval_nll(&self, params: &[f64]) -> f64 {
    let clamped = self.clamp(params);
    let mut model = self.model.lock();
    model.set_params(&clamped);
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

  fn cost(&self, params: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
    Ok(self.eval_nll(params))
  }
}

impl Gradient for MleProblem<'_> {
  type Param = Vec<f64>;
  type Gradient = Vec<f64>;

  fn gradient(&self, params: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
    let clamped = self.clamp(params);
    let n = clamped.len();
    let mut grad = vec![0.0; n];
    for i in 0..n {
      let h = 1e-7 * (1.0 + clamped[i].abs());
      let mut p_plus = clamped.clone();
      let mut p_minus = clamped.clone();
      p_plus[i] = (clamped[i] + h).min(self.bounds[i].1);
      p_minus[i] = (clamped[i] - h).max(self.bounds[i].0);
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

/// Resolves an L-BFGS [`Executor`] outcome into the parameter vector
/// `fit_mle` should keep, plus the `converged` / `iterations` signals on
/// [`MleResult`].
///
/// `Err` means the optimiser itself failed before completing a step:
/// `init` is returned untouched and `converged` is `false` so callers can
/// never mistake the fallback for a genuine fit (this is the case
/// formerly handled by a silent `Err(_) => init`, with no signal that a
/// fallback had occurred). On `Ok`, `converged` additionally requires
/// argmin to report `SolverConverged` or `TargetCostReached` —
/// `MaxItersReached`, `SolverExit` (e.g. an internal line-search
/// failure), `Timeout` and `Interrupt` all mean the run stopped without
/// meeting its own tolerance, so they also signal `false` even though
/// the `Executor` itself did not error.
fn resolve_fit_outcome(
  result: Result<(Vec<f64>, TerminationStatus, u64), argmin::core::Error>,
  init: &[f64],
) -> (Vec<f64>, bool, usize) {
  match result {
    Ok((best_param, status, iters)) => {
      let converged = matches!(
        status,
        TerminationStatus::Terminated(TerminationReason::SolverConverged)
          | TerminationStatus::Terminated(TerminationReason::TargetCostReached)
      );
      (best_param, converged, iters as usize)
    }
    Err(_) => (init.to_vec(), false, 0),
  }
}

/// Fit a 1-D SDE model by Maximum Likelihood Estimation.
///
/// The function minimises the negative log-likelihood
///
/// $$
/// -\sum_{i=1}^{N} \ln p(X_{t_i}\mid X_{t_{i-1}};\theta,\Delta t)
/// $$
///
/// using L-BFGS (via argmin) with numerical gradient and projected box constraints.
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
/// `converged` before trusting `params` — a failed or non-converged run
/// returns the initial guess rather than panicking or fabricating a fit.
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

    let problem = MleProblem {
      model: Mutex::new(&mut *model),
      sample,
      dt,
      density,
      bounds: bounds.clone(),
    };

    let linesearch = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(linesearch, 10);

    let result = Executor::new(problem, solver)
      .configure(|state| state.param(init.clone()).max_iters(200))
      .run()
      .map(|res| {
        let best = res
          .state
          .get_best_param()
          .cloned()
          .unwrap_or_else(|| init.clone());
        let status = res.state.get_termination_status().clone();
        let iters = res.state.get_iter();
        (best, status, iters)
      });

    resolve_fit_outcome(result, &init)
  };

  let clamped: Vec<f64> = best_params
    .iter()
    .enumerate()
    .map(|(i, &x)| x.clamp(bounds[i].0, bounds[i].1))
    .collect();

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
  use super::*;

  #[test]
  fn mle_result_signals_fallback() {
    let init = vec![1.0, 2.0, 3.0];
    let forced_failure: Result<(Vec<f64>, TerminationStatus, u64), argmin::core::Error> =
      Err(argmin::core::Error::msg("forced optimizer failure"));

    let (params, converged, iterations) = resolve_fit_outcome(forced_failure, &init);

    assert_eq!(
      params, init,
      "a failed optimizer run must return the untouched initial guess"
    );
    assert!(
      !converged,
      "a failed optimizer run must never silently report converged = true"
    );
    assert_eq!(iterations, 0);
  }

  #[test]
  fn mle_result_signals_non_convergence_without_error() {
    // MaxItersReached (and SolverExit / Timeout / Interrupt) is a normal
    // `Ok` termination from argmin's perspective, not an `Err` — but it
    // still means the run never actually converged, so it must signal
    // `converged = false` exactly like the `Err` branch.
    let init = vec![0.5];
    let stalled: Result<(Vec<f64>, TerminationStatus, u64), argmin::core::Error> = Ok((
      init.clone(),
      TerminationStatus::Terminated(TerminationReason::MaxItersReached),
      200,
    ));

    let (params, converged, iterations) = resolve_fit_outcome(stalled, &init);

    assert_eq!(params, init);
    assert!(
      !converged,
      "MaxItersReached must not be reported as converged"
    );
    assert_eq!(iterations, 200);
  }

  #[test]
  fn mle_result_signals_genuine_convergence() {
    let fitted = vec![1.23, 4.56];
    let ok: Result<(Vec<f64>, TerminationStatus, u64), argmin::core::Error> = Ok((
      fitted.clone(),
      TerminationStatus::Terminated(TerminationReason::SolverConverged),
      17,
    ));

    let (params, converged, iterations) = resolve_fit_outcome(ok, &[0.0, 0.0]);

    assert_eq!(params, fitted);
    assert!(converged);
    assert_eq!(iterations, 17);
  }
}
