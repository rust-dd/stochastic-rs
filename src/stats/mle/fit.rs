use std::collections::VecDeque;
use std::fmt;

use ndarray::Array1;

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
    Ok(())
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
/// using L-BFGS with numerical gradient and projected box constraints.
///
/// # Arguments
/// * `model`        - the SDE model (parameters will be set to the MLE values on return)
/// * `sample`       - observed sample path (length N+1)
/// * `dt`           - sampling interval
/// * `density`      - transition density approximation method
/// * `param_bounds` - optional custom bounds (defaults to model's `param_bounds()`)
///
/// # Returns
/// An [`MleResult`] with estimated parameters, log-likelihood, AIC and BIC.
///
/// # References
/// - Nocedal, J. (1980). *Mathematics of Computation*, 35(151), 773-782.
///   <https://doi.org/10.1090/S0025-5718-1980-0572855-7>
/// - Liu, D.C. & Nocedal, J. (1989). *Mathematical Programming*, 45, 503-528.
///   <https://doi.org/10.1007/BF01589116>
pub fn fit_mle(
  model: &mut dyn DiffusionModel,
  sample: &Array1<f64>,
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

  let best_params = if n_params == 0 {
    x0
  } else {
    lbfgs_mle(model, sample, dt, &density, &x0, &bounds)
  };

  model.set_params(best_params.as_slice().unwrap());
  let log_lik = -eval_nll(model, &best_params, sample, dt, &density);

  let k = n_params as f64;
  let n = n_transitions as f64;
  let aic = 2.0 * k - 2.0 * log_lik;
  let bic = k * n.ln() - 2.0 * log_lik;

  MleResult {
    params: best_params,
    param_names: model.param_names().into_iter().map(String::from).collect(),
    log_likelihood: log_lik,
    sample_size: n_transitions,
    aic,
    bic,
  }
}

/// Evaluate negative log-likelihood.
fn eval_nll(
  model: &mut dyn DiffusionModel,
  params: &Array1<f64>,
  sample: &Array1<f64>,
  dt: f64,
  density: &DensityApprox,
) -> f64 {
  model.set_params(params.as_slice().unwrap());
  let mut sum = 0.0;
  for i in 1..sample.len() {
    let t0 = (i - 1) as f64 * dt;
    let d = density.density(model, sample[i - 1], sample[i], t0, dt);
    sum -= d.max(1e-30).ln();
  }
  if sum.is_finite() { sum } else { 1e30 }
}

/// Numerical gradient via central finite differences.
fn numerical_gradient(
  model: &mut dyn DiffusionModel,
  params: &Array1<f64>,
  sample: &Array1<f64>,
  dt: f64,
  density: &DensityApprox,
  bounds: &[(f64, f64)],
) -> Array1<f64> {
  let n = params.len();
  let mut grad = Array1::zeros(n);
  for i in 0..n {
    let h = 1e-7 * (1.0 + params[i].abs());
    let mut p_plus = params.clone();
    let mut p_minus = params.clone();
    p_plus[i] = (params[i] + h).min(bounds[i].1);
    p_minus[i] = (params[i] - h).max(bounds[i].0);
    let actual_2h = p_plus[i] - p_minus[i];
    if actual_2h > 0.0 {
      let f_plus = eval_nll(model, &p_plus, sample, dt, density);
      let f_minus = eval_nll(model, &p_minus, sample, dt, density);
      grad[i] = (f_plus - f_minus) / actual_2h;
    }
  }
  grad
}

/// Project vector onto box constraints.
fn clamp_to_bounds(v: &Array1<f64>, bounds: &[(f64, f64)]) -> Array1<f64> {
  Array1::from_vec(
    v.iter()
      .enumerate()
      .map(|(i, &x)| x.clamp(bounds[i].0, bounds[i].1))
      .collect(),
  )
}

/// L-BFGS two-loop recursion to compute the search direction d = -H·g.
///
/// Ref: Nocedal & Wright, *Numerical Optimization*, 2nd ed., Algorithm 7.4.
fn lbfgs_direction(
  g: &Array1<f64>,
  s_hist: &VecDeque<Array1<f64>>,
  y_hist: &VecDeque<Array1<f64>>,
  rho_hist: &VecDeque<f64>,
) -> Array1<f64> {
  let k = s_hist.len();
  if k == 0 {
    return -g.clone();
  }

  let mut q = g.clone();
  let mut alphas = vec![0.0; k];

  for i in (0..k).rev() {
    alphas[i] = rho_hist[i] * s_hist[i].dot(&q);
    q = &q - &(alphas[i] * &y_hist[i]);
  }

  let gamma = s_hist[k - 1].dot(&y_hist[k - 1]) / y_hist[k - 1].dot(&y_hist[k - 1]);
  let mut r = gamma * &q;

  for i in 0..k {
    let beta = rho_hist[i] * y_hist[i].dot(&r);
    r = &r + &((alphas[i] - beta) * &s_hist[i]);
  }

  -r
}

/// L-BFGS optimiser with projected box constraints and backtracking line search.
fn lbfgs_mle(
  model: &mut dyn DiffusionModel,
  sample: &Array1<f64>,
  dt: f64,
  density: &DensityApprox,
  x0: &Array1<f64>,
  bounds: &[(f64, f64)],
) -> Array1<f64> {
  let m = 10;
  let max_iter = 200;
  let grad_tol = 1e-8;
  let f_tol = 1e-12;
  let c1 = 1e-4;

  let mut x = clamp_to_bounds(x0, bounds);
  let mut f = eval_nll(model, &x, sample, dt, density);
  let mut g = numerical_gradient(model, &x, sample, dt, density, bounds);

  let mut s_hist: VecDeque<Array1<f64>> = VecDeque::new();
  let mut y_hist: VecDeque<Array1<f64>> = VecDeque::new();
  let mut rho_hist: VecDeque<f64> = VecDeque::new();

  let mut best_x = x.clone();
  let mut best_f = f;

  for _ in 0..max_iter {
    let g_norm = g.dot(&g).sqrt();
    if g_norm < grad_tol {
      break;
    }

    let mut d = lbfgs_direction(&g, &s_hist, &y_hist, &rho_hist);

    if g.dot(&d) >= 0.0 {
      d = -&g;
      s_hist.clear();
      y_hist.clear();
      rho_hist.clear();
    }

    let gd = g.dot(&d);

    // Backtracking line search (Armijo condition)
    let mut alpha = 1.0;
    let mut success = false;

    for _ in 0..30 {
      let x_new = clamp_to_bounds(&(&x + &(alpha * &d)), bounds);
      let f_new = eval_nll(model, &x_new, sample, dt, density);

      if f_new <= f + c1 * alpha * gd {
        let g_new = numerical_gradient(model, &x_new, sample, dt, density, bounds);
        let s = &x_new - &x;
        let y = &g_new - &g;
        let sy = s.dot(&y);

        if sy > 1e-10 {
          if s_hist.len() >= m {
            s_hist.pop_front();
            y_hist.pop_front();
            rho_hist.pop_front();
          }
          s_hist.push_back(s);
          y_hist.push_back(y);
          rho_hist.push_back(1.0 / sy);
        }

        if f_new < best_f {
          best_x = x_new.clone();
          best_f = f_new;
        }

        let f_change = (f - f_new).abs();
        x = x_new;
        f = f_new;
        g = g_new;
        success = true;

        if f_change < f_tol {
          return best_x;
        }
        break;
      }
      alpha *= 0.5;
    }

    if !success {
      break;
    }
  }

  best_x
}
