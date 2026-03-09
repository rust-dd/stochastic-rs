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
/// using bounded Nelder-Mead optimisation.
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
/// Source: pymle (<https://github.com/jkirkby3/pymle>)
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

  let nll = |params: &Array1<f64>, model: &mut dyn DiffusionModel| -> f64 {
    model.set_params(params.as_slice().unwrap());
    let mut sum = 0.0;
    for i in 1..sample.len() {
      let t0 = (i - 1) as f64 * dt;
      let d = density.density(model, sample[i - 1], sample[i], t0, dt);
      sum -= d.max(1e-30).ln();
    }
    if sum.is_finite() { sum } else { 1e30 }
  };

  let n_restarts = 3;
  let mut best_params = x0.clone();
  let mut best_nll = nll(&best_params, model);

  for restart in 0..n_restarts {
    let max_iter = if restart == 0 { 10_000 } else { 5_000 };
    let init_scale = if restart == 0 { 0.15 } else { 0.05 };
    let candidate = nelder_mead_mle(
      model,
      sample,
      dt,
      &density,
      &best_params,
      &bounds,
      max_iter,
      1e-12,
      init_scale,
    );
    let candidate_nll = nll(&candidate, model);
    if candidate_nll < best_nll {
      best_params = candidate;
      best_nll = candidate_nll;
    }
  }

  model.set_params(best_params.as_slice().unwrap());
  let log_lik = -best_nll;

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

/// Nelder-Mead simplex optimiser with box constraints.
fn nelder_mead_mle(
  model: &mut dyn DiffusionModel,
  sample: &Array1<f64>,
  dt: f64,
  density: &DensityApprox,
  x0: &Array1<f64>,
  bounds: &[(f64, f64)],
  max_iter: usize,
  tol: f64,
  init_scale: f64,
) -> Array1<f64> {
  let n = x0.len();

  let eval = |params: &Array1<f64>, mdl: &mut dyn DiffusionModel| -> f64 {
    mdl.set_params(params.as_slice().unwrap());
    let mut sum = 0.0;
    for i in 1..sample.len() {
      let t0 = (i - 1) as f64 * dt;
      let d = density.density(mdl, sample[i - 1], sample[i], t0, dt);
      sum -= d.max(1e-30).ln();
    }
    if sum.is_finite() { sum } else { 1e30 }
  };

  let clamp = |x: f64, b: (f64, f64)| -> f64 { x.clamp(b.0, b.1) };

  let clamp_arr = |v: &Array1<f64>| -> Array1<f64> {
    Array1::from_vec(
      v.iter()
        .enumerate()
        .map(|(i, &x)| clamp(x, bounds[i]))
        .collect(),
    )
  };

  let alpha = 1.0;
  let gamma_nm = 2.0;
  let rho = 0.5;
  let sigma_nm = 0.5;

  let mut simplex: Vec<Array1<f64>> = Vec::with_capacity(n + 1);
  simplex.push(clamp_arr(x0));
  for i in 0..n {
    let mut point = clamp_arr(x0);
    let delta = if x0[i].abs() > 1e-10 {
      init_scale * x0[i].abs()
    } else {
      0.00025 * (1.0 + init_scale * 100.0)
    };
    point[i] = clamp(point[i] + delta, bounds[i]);
    let base_i = clamp(x0[i], bounds[i]);
    if (point[i] - base_i).abs() < 1e-15 {
      point[i] = clamp(base_i - delta, bounds[i]);
    }
    simplex.push(point);
  }

  let mut values: Vec<f64> = simplex.iter().map(|p| eval(p, model)).collect();

  for _iter in 0..max_iter {
    let mut order: Vec<usize> = (0..=n).collect();
    order.sort_by(|&a, &b| {
      values[a]
        .partial_cmp(&values[b])
        .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_simplex: Vec<Array1<f64>> = order.iter().map(|&i| simplex[i].clone()).collect();
    let sorted_values: Vec<f64> = order.iter().map(|&i| values[i]).collect();
    simplex = sorted_simplex;
    values = sorted_values;

    if (values[n] - values[0]).abs() < tol {
      break;
    }

    let mut centroid = Array1::<f64>::zeros(n);
    for j in 0..n {
      for i in 0..n {
        centroid[j] += simplex[i][j];
      }
      centroid[j] /= n as f64;
    }

    let reflected = Array1::from_vec(
      (0..n)
        .map(|j| clamp(centroid[j] + alpha * (centroid[j] - simplex[n][j]), bounds[j]))
        .collect(),
    );
    let f_reflected = eval(&reflected, model);

    if f_reflected < values[0] {
      let expanded = Array1::from_vec(
        (0..n)
          .map(|j| {
            clamp(
              centroid[j] + gamma_nm * (reflected[j] - centroid[j]),
              bounds[j],
            )
          })
          .collect(),
      );
      let f_expanded = eval(&expanded, model);
      if f_expanded < f_reflected {
        simplex[n] = expanded;
        values[n] = f_expanded;
      } else {
        simplex[n] = reflected;
        values[n] = f_reflected;
      }
    } else if f_reflected < values[n - 1] {
      simplex[n] = reflected;
      values[n] = f_reflected;
    } else {
      let contracted = Array1::from_vec(
        (0..n)
          .map(|j| clamp(centroid[j] + rho * (simplex[n][j] - centroid[j]), bounds[j]))
          .collect(),
      );
      let f_contracted = eval(&contracted, model);
      if f_contracted < values[n] {
        simplex[n] = contracted;
        values[n] = f_contracted;
      } else {
        let best = simplex[0].clone();
        for i in 1..=n {
          for j in 0..n {
            simplex[i][j] = clamp(best[j] + sigma_nm * (simplex[i][j] - best[j]), bounds[j]);
          }
          values[i] = eval(&simplex[i], model);
        }
      }
    }
  }

  let best_idx = values
    .iter()
    .enumerate()
    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
    .unwrap()
    .0;
  simplex[best_idx].clone()
}
