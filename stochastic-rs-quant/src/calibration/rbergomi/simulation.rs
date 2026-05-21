use std::sync::Arc;

use nalgebra::DMatrix;
use rayon::prelude::*;
use stochastic_rs_distributions::special::gamma;
use stochastic_rs_distributions::special::gamma_li;

use super::XI0_MIN;
use super::params::RBergomiParams;

#[derive(Clone)]
pub(super) struct MsoeEngine {
  pub(super) h: f64,
  pub(super) dt: f64,
  pub(super) lambdas: Vec<f64>,
  pub(super) weights: Vec<f64>,
  pub(super) decay: Vec<f64>,
  pub(super) second_moment: Vec<f64>,
  pub(super) chol_l: ndarray::Array2<f64>,
}

impl MsoeEngine {
  pub(super) fn new(h: f64, dt: f64, maturity: f64, steps: usize, terms: usize) -> Self {
    let (lambdas, weights) = build_msoe_kernel(h, dt, maturity, terms.max(2));
    let decay = lambdas.iter().map(|x| (-x * dt).exp()).collect::<Vec<_>>();
    let cov = build_step_covariance(h, dt, &lambdas);
    let l = cholesky_lower_with_jitter(cov);
    let mut chol_l = ndarray::Array2::<f64>::zeros((l.nrows(), l.ncols()));
    for row in 0..l.nrows() {
      for col in 0..=row {
        chol_l[(row, col)] = l[(row, col)];
      }
    }
    let second_moment = precompute_second_moments(h, dt, steps, &lambdas, &weights);

    Self {
      h,
      dt,
      lambdas,
      weights,
      decay,
      second_moment,
      chol_l,
    }
  }

  pub(super) fn dim(&self) -> usize {
    self.lambdas.len() + 2
  }

  pub(super) fn terms(&self) -> usize {
    self.lambdas.len()
  }

  pub(super) fn transform(&self, z: &[f64], out: &mut [f64]) {
    debug_assert_eq!(z.len(), self.dim());
    debug_assert_eq!(out.len(), self.dim());
    for (idx, item) in out.iter_mut().enumerate().take(self.dim()) {
      let mut acc = 0.0;
      for col in 0..=idx {
        acc += self.chol_l[(idx, col)] * z[col];
      }
      *item = acc;
    }
  }
}

/// Simulates terminal prices under the rBergomi model with an mSOE approximation
/// for the Volterra kernel history term.
///
/// The log-stock drift is `(r - q - 0.5·V_t) dt`. The `q` argument is the
/// continuously-compounded dividend yield (or foreign rate for FX); callers
/// who don't pay dividends should pass `0.0`.
#[allow(clippy::too_many_arguments)]
pub fn simulate_rbergomi_terminal_samples(
  params: &RBergomiParams,
  s0: f64,
  r: f64,
  q: f64,
  maturity: f64,
  paths: usize,
  steps_per_year: usize,
  msoe_terms: usize,
  seed: u64,
) -> Vec<f64> {
  assert!(
    maturity.is_finite() && maturity > 0.0,
    "maturity must be > 0"
  );
  assert!(paths > 0, "paths must be > 0");
  assert!(steps_per_year > 0, "steps_per_year must be > 0");

  let params = params.clone().projected();
  let steps = ((maturity * steps_per_year as f64).ceil() as usize).max(2);
  let dt = maturity / steps as f64;
  let sqrt_dt = dt.sqrt();

  let engine = Arc::new(MsoeEngine::new(
    params.hurst,
    dt,
    maturity,
    steps,
    msoe_terms.max(2),
  ));
  let rho = params.rho;
  let rho_orth = (1.0 - rho * rho).max(0.0).sqrt();

  (0..paths)
    .into_par_iter()
    .map(|path_idx| {
      let path_seed = seed
        .wrapping_add(0xD134_2543_DE82_EF95_u64.wrapping_mul((path_idx as u64).wrapping_add(1)));
      let seed_ext = crate::simd_rng::Deterministic::new(path_seed);
      let normal = crate::distributions::normal::SimdNormal::<f64>::new(0.0, 1.0, &seed_ext);
      let dim = engine.dim();
      let mut z = vec![0.0_f64; dim];
      let mut xi = vec![0.0_f64; dim];
      let mut history = vec![0.0_f64; engine.terms()];

      let mut s = s0.max(1e-12);
      let mut v_prev = params.xi0.value(0.0).max(XI0_MIN);

      for step in 1..=steps {
        for zi in z.iter_mut() {
          *zi = normal.sample_fast();
        }
        engine.transform(&z, &mut xi);

        let d_w = xi[0];
        let d_w_perp = normal.sample_fast() * sqrt_dt;

        let drift = (r - q - 0.5 * v_prev) * dt;
        let diffusion = v_prev.sqrt() * (rho * d_w + rho_orth * d_w_perp);
        s *= (drift + diffusion).exp();

        let mut past_sum = 0.0;
        for (idx, item) in history.iter().enumerate().take(engine.terms()) {
          past_sum += engine.weights[idx] * item;
        }

        let i_hat = xi[dim - 1] + (2.0 * engine.h).sqrt() * past_sum;
        let t = step as f64 * engine.dt;
        let forward_var = params.xi0.value(t).max(XI0_MIN);
        let second_moment = engine.second_moment[step - 1].max(1e-14);
        let v_new =
          forward_var * (params.eta * i_hat - 0.5 * params.eta * params.eta * second_moment).exp();
        v_prev = v_new.max(XI0_MIN);

        for k in 0..engine.terms() {
          history[k] = engine.decay[k] * (history[k] + xi[1 + k]);
        }
      }

      s
    })
    .collect()
}

fn build_msoe_kernel(h: f64, dt: f64, maturity: f64, terms: usize) -> (Vec<f64>, Vec<f64>) {
  let terms = terms.max(2);
  let gamma_norm = gamma(0.5 - h);

  let x_min = ((1.0 / maturity.max(dt)) * 1e-2).max(1e-8);
  let x_max = ((1.0 / dt.max(1e-8)) * 50.0).max(x_min * 10.0);
  let y_min = x_min.ln();
  let y_max = x_max.ln();
  let dy = (y_max - y_min) / (terms as f64 - 1.0);

  let mut lambdas = Vec::with_capacity(terms);
  let mut weights = Vec::with_capacity(terms);

  for j in 0..terms {
    let y = y_min + j as f64 * dy;
    let x = y.exp();
    let boundary = if j == 0 || j + 1 == terms { 0.5 } else { 1.0 };
    let w = boundary * dy * ((0.5 - h) * y).exp() / gamma_norm;
    lambdas.push(x);
    weights.push(w.max(0.0));
  }

  (lambdas, weights)
}

fn build_step_covariance(h: f64, dt: f64, lambdas: &[f64]) -> DMatrix<f64> {
  let n = lambdas.len();
  let dim = n + 2;
  let mut sigma = DMatrix::<f64>::zeros(dim, dim);
  let local_idx = dim - 1;

  sigma[(0, 0)] = dt;

  for (k, lambda) in lambdas.iter().enumerate() {
    let idx = k + 1;
    let cov = (1.0 - (-lambda * dt).exp()) / lambda;
    sigma[(0, idx)] = cov;
    sigma[(idx, 0)] = cov;
  }

  for (k, lambda_k) in lambdas.iter().enumerate() {
    for (l, lambda_l) in lambdas.iter().enumerate() {
      let idx_k = k + 1;
      let idx_l = l + 1;
      let sum = lambda_k + lambda_l;
      sigma[(idx_k, idx_l)] = (1.0 - (-sum * dt).exp()) / sum;
    }
  }

  let cov_local_dw = (2.0 * h).sqrt() / (h + 0.5) * dt.powf(h + 0.5);
  sigma[(local_idx, 0)] = cov_local_dw;
  sigma[(0, local_idx)] = cov_local_dw;

  for (k, lambda) in lambdas.iter().enumerate() {
    let idx = k + 1;
    let a = h + 0.5;
    let cov = (2.0 * h).sqrt() * lambda.powf(-a) * gamma_li(a, lambda * dt);
    sigma[(local_idx, idx)] = cov;
    sigma[(idx, local_idx)] = cov;
  }

  sigma[(local_idx, local_idx)] = dt.powf(2.0 * h);
  sigma
}

fn cholesky_lower_with_jitter(mut sigma: DMatrix<f64>) -> DMatrix<f64> {
  let dim = sigma.nrows();
  let mut jitter = 1e-12;
  for _ in 0..8 {
    if let Some(chol) = sigma.clone().cholesky() {
      return chol.l();
    }
    for i in 0..dim {
      sigma[(i, i)] += jitter;
    }
    jitter *= 10.0;
  }

  // Conservative fallback: keep marginal variances, drop correlations.
  let mut l = DMatrix::<f64>::zeros(dim, dim);
  for i in 0..dim {
    l[(i, i)] = sigma[(i, i)].max(1e-14).sqrt();
  }
  l
}

fn precompute_second_moments(
  h: f64,
  dt: f64,
  steps: usize,
  lambdas: &[f64],
  weights: &[f64],
) -> Vec<f64> {
  let n = lambdas.len();
  let local_var = dt.powf(2.0 * h);
  let mut out = vec![0.0_f64; steps];

  for i in 1..=steps {
    let t_i = i as f64 * dt;
    let mut v = local_var;
    for k in 0..n {
      for l in 0..n {
        let a = lambdas[k] + lambdas[l];
        let coeff = 2.0 * h * weights[k] * weights[l] / a;
        v += coeff * ((-a * dt).exp() - (-a * t_i).exp());
      }
    }
    out[i - 1] = v.max(1e-14);
  }

  out
}
