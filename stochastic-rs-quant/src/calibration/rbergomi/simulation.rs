use std::sync::Arc;

use ndarray::Array2;
use rayon::prelude::*;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::linalg::spd_cholesky_lower;
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
      *item = z[..=idx]
        .iter()
        .enumerate()
        .fold(0.0, |acc, (col, z)| acc + self.chol_l[(idx, col)] * z);
    }
  }
}

/// Terminal prices by the mSOE kernel under the log drift `(r − q − V_t/2) dt`: `s0` at zero
/// maturity, NaN for a NaN or negative one, a NaN rate or a spot not positive; panics on no path.
// The model, the query, the grid's size and the seed.
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
  seed: &impl SeedExt,
) -> Vec<f64> {
  let query = (s0, r, q, maturity);
  let mirrored = simulate_mirrored(
    params,
    query,
    paths,
    steps_per_year,
    msoe_terms,
    seed,
    [1.0],
  );
  mirrored.into_iter().map(|[s]| s).collect()
}

/// Antithetic pairs of terminal prices: each path's second leg runs on its first leg's normals
/// negated, the Volterra driver's and the orthogonal Brownian's alike.
pub(crate) fn simulate_rbergomi_antithetic_pairs(
  params: &RBergomiParams,
  query: (f64, f64, f64, f64),
  paths: usize,
  steps_per_year: usize,
  msoe_terms: usize,
  seed: &impl SeedExt,
) -> Vec<[f64; 2]> {
  simulate_mirrored(
    params,
    query,
    paths,
    steps_per_year,
    msoe_terms,
    seed,
    [1.0, -1.0],
  )
}

/// One terminal price per sign for each path, every leg of a path on the same normals times its
/// sign; the query is `(s0, r, q, maturity)`.
fn simulate_mirrored<const N: usize>(
  params: &RBergomiParams,
  (s0, r, q, maturity): (f64, f64, f64, f64),
  paths: usize,
  steps_per_year: usize,
  msoe_terms: usize,
  seed: &impl SeedExt,
  signs: [f64; N],
) -> Vec<[f64; N]> {
  assert!(
    paths > 0,
    "paths must satisfy `paths > 0`, got paths = {paths:?}"
  );
  assert!(
    steps_per_year > 0,
    "steps_per_year must satisfy `steps_per_year > 0`, got steps_per_year = {steps_per_year:?}"
  );
  let defined = s0 > 0.0 && maturity >= 0.0 && maturity.is_finite() && (r - q).is_finite();
  if !defined || maturity == 0.0 {
    return vec![[if defined { s0 } else { f64::NAN }; N]; paths];
  }

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
  let seed = seed.next_seed();
  let xi0_start = params.xi0.value(0.0).max(XI0_MIN);

  (0..paths)
    .into_par_iter()
    .map(|path_idx| {
      let path_seed = seed
        .wrapping_add(0xD134_2543_DE82_EF95_u64.wrapping_mul((path_idx as u64).wrapping_add(1)));
      let seed_ext = Deterministic::new(path_seed);
      let mut normal =
        stochastic_rs_distributions::normal::SimdNormal::<f64>::new(0.0, 1.0).seeded(&seed_ext);
      let dim = engine.dim();
      let mut z = vec![0.0_f64; dim];
      let mut xi = vec![0.0_f64; dim];
      let mut history = [(); N].map(|_| vec![0.0_f64; engine.terms()]);
      let mut s = [s0; N];
      let mut v_prev = [xi0_start; N];

      for step in 1..=steps {
        for zi in z.iter_mut() {
          *zi = normal.sample();
        }
        engine.transform(&z, &mut xi);
        let d_w_perp = normal.sample() * sqrt_dt;
        let t = step as f64 * engine.dt;
        let forward_var = params.xi0.value(t).max(XI0_MIN);
        let second_moment = engine.second_moment[step - 1].max(1e-14);

        for (leg, &sign) in signs.iter().enumerate() {
          let drift = (r - q - 0.5 * v_prev[leg]) * dt;
          let shock = rho * (sign * xi[0]) + rho_orth * (sign * d_w_perp);
          s[leg] *= (drift + v_prev[leg].sqrt() * shock).exp();

          let mut past_sum = 0.0;
          for (idx, item) in history[leg].iter().enumerate().take(engine.terms()) {
            past_sum += engine.weights[idx] * item;
          }
          let i_hat = sign * xi[dim - 1] + (2.0 * engine.h).sqrt() * past_sum;
          let v_new = forward_var
            * (params.eta * i_hat - 0.5 * params.eta * params.eta * second_moment).exp();
          v_prev[leg] = v_new.max(XI0_MIN);

          for k in 0..engine.terms() {
            history[leg][k] = engine.decay[k] * (history[leg][k] + sign * xi[1 + k]);
          }
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

fn build_step_covariance(h: f64, dt: f64, lambdas: &[f64]) -> Array2<f64> {
  let n = lambdas.len();
  let dim = n + 2;
  let mut sigma = Array2::<f64>::zeros((dim, dim));
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

fn cholesky_lower_with_jitter(mut sigma: Array2<f64>) -> Array2<f64> {
  let dim = sigma.nrows();
  let mut jitter = 1e-12;
  for _ in 0..8 {
    if let Some(lower) = spd_cholesky_lower(sigma.view()) {
      return lower;
    }
    for i in 0..dim {
      sigma[(i, i)] += jitter;
    }
    jitter *= 10.0;
  }

  // Conservative fallback: keep marginal variances, drop correlations.
  let mut l = Array2::<f64>::zeros((dim, dim));
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
