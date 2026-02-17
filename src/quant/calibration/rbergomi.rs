//! # rBergomi Calibration
//!
//! rBergomi dynamics under the risk-neutral measure:
//! $$
//! dS_t=rS_t\,dt+S_t\sqrt{V_t}\left(\rho\,dW_t+\sqrt{1-\rho^2}\,dW_t^\perp\right),
//! $$
//! $$
//! V_t=\xi_0(t)\exp\left(\eta I_t-\frac{\eta^2}{2}t^{2H}\right),\quad
//! I_t=\sqrt{2H}\int_0^t (t-s)^{H-\frac12}\,dW_s.
//! $$
//!
//! Calibration objective (distribution matching):
//! $$
//! L(\theta)=\frac1M\sum_{j=1}^M W_1\left(S_{T_j}(\theta),S_{T_j}^{\mathrm{MKT}}\right).
//! $$
//!
//! Empirical Wasserstein-1 in 1D:
//! $$
//! W_1\approx\frac1m\sum_{i=1}^m\left|X_{(i)}-Y_{(i)}\right|.
//! $$
//!
//! Source:
//! - Rough Bergomi model: https://arxiv.org/abs/1609.02108
//! - Wasserstein calibration and mSOE-style simulation formulas.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::Distribution;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use statrs::function::gamma::gamma;
use statrs::function::gamma::gamma_li;

const H_MIN: f64 = 1e-3;
const H_MAX: f64 = 0.499;
const RHO_BOUND: f64 = 0.999;
const ETA_MIN: f64 = 1e-8;
const XI0_MIN: f64 = 1e-8;

#[derive(Clone, Debug)]
pub enum RBergomiXi0 {
  /// $\xi_0(t)=\theta_0$
  Constant(f64),
  /// $\xi_0(t)=\sum_{l=1}^L \theta_l \mathbf{1}_{[T_{l-1},T_l)}(t)$
  PiecewiseConstant {
    maturities: Vec<f64>,
    values: Vec<f64>,
  },
  /// $\xi_0(t)=\beta_0+\beta_1e^{-t/\tau}+\beta_2(t/\tau)e^{-t/\tau}$
  NelsonSiegel {
    beta0: f64,
    beta1: f64,
    beta2: f64,
    tau: f64,
  },
}

impl RBergomiXi0 {
  pub fn value(&self, t: f64) -> f64 {
    match self {
      Self::Constant(level) => level.max(XI0_MIN),
      Self::PiecewiseConstant { maturities, values } => {
        if maturities.len() < 2 || values.is_empty() {
          return XI0_MIN;
        }
        if t < maturities[0] {
          return values[0].max(XI0_MIN);
        }
        for i in 1..maturities.len() {
          if t < maturities[i] {
            return values[i - 1].max(XI0_MIN);
          }
        }
        values[values.len() - 1].max(XI0_MIN)
      }
      Self::NelsonSiegel {
        beta0,
        beta1,
        beta2,
        tau,
      } => {
        let tau = tau.abs().max(1e-6);
        let x = (t.max(0.0)) / tau;
        (beta0 + beta1 * (-x).exp() + beta2 * x * (-x).exp()).max(XI0_MIN)
      }
    }
  }

  fn project_in_place(&mut self) {
    match self {
      Self::Constant(level) => {
        *level = level.abs().max(XI0_MIN);
      }
      Self::PiecewiseConstant { values, .. } => {
        for v in values.iter_mut() {
          *v = v.abs().max(XI0_MIN);
        }
      }
      Self::NelsonSiegel { tau, .. } => {
        *tau = tau.abs().max(1e-6);
      }
    }
  }

  fn flattened_len(&self) -> usize {
    match self {
      Self::Constant(_) => 1,
      Self::PiecewiseConstant { values, .. } => values.len(),
      Self::NelsonSiegel { .. } => 4,
    }
  }

  fn flatten_into(&self, out: &mut Vec<f64>) {
    match self {
      Self::Constant(level) => out.push(*level),
      Self::PiecewiseConstant { values, .. } => out.extend(values.iter().copied()),
      Self::NelsonSiegel {
        beta0,
        beta1,
        beta2,
        tau,
      } => {
        out.push(*beta0);
        out.push(*beta1);
        out.push(*beta2);
        out.push(*tau);
      }
    }
  }

  fn assign_from_flattened(&mut self, values: &[f64], offset: &mut usize) {
    match self {
      Self::Constant(level) => {
        *level = values[*offset];
        *offset += 1;
      }
      Self::PiecewiseConstant { values: out, .. } => {
        let end = *offset + out.len();
        out.copy_from_slice(&values[*offset..end]);
        *offset = end;
      }
      Self::NelsonSiegel {
        beta0,
        beta1,
        beta2,
        tau,
      } => {
        *beta0 = values[*offset];
        *beta1 = values[*offset + 1];
        *beta2 = values[*offset + 2];
        *tau = values[*offset + 3];
        *offset += 4;
      }
    }
  }

  fn validate(&self) -> Result<(), String> {
    match self {
      Self::Constant(level) => {
        if !level.is_finite() || *level <= 0.0 {
          return Err("RBergomiXi0::Constant must be finite and positive".to_string());
        }
      }
      Self::PiecewiseConstant { maturities, values } => {
        if maturities.len() < 2 {
          return Err(
            "RBergomiXi0::PiecewiseConstant requires at least two maturity pillars".to_string(),
          );
        }
        if values.len() + 1 != maturities.len() {
          return Err(
            "RBergomiXi0::PiecewiseConstant requires values.len() + 1 == maturities.len()"
              .to_string(),
          );
        }
        if !maturities.windows(2).all(|w| w[0] < w[1]) {
          return Err(
            "RBergomiXi0::PiecewiseConstant maturities must be strictly increasing".to_string(),
          );
        }
        if values.iter().any(|v| !v.is_finite() || *v <= 0.0) {
          return Err(
            "RBergomiXi0::PiecewiseConstant values must be finite and positive".to_string(),
          );
        }
      }
      Self::NelsonSiegel {
        beta0,
        beta1,
        beta2,
        tau,
      } => {
        if !beta0.is_finite() || !beta1.is_finite() || !beta2.is_finite() || !tau.is_finite() {
          return Err("RBergomiXi0::NelsonSiegel parameters must be finite".to_string());
        }
        if *tau <= 0.0 {
          return Err("RBergomiXi0::NelsonSiegel tau must be positive".to_string());
        }
      }
    }
    Ok(())
  }
}

#[derive(Clone, Debug)]
pub struct RBergomiParams {
  /// Hurst exponent in $(0,\frac12)$ for rough volatility.
  pub hurst: f64,
  /// Instantaneous leverage correlation.
  pub rho: f64,
  /// Vol-of-vol parameter.
  pub eta: f64,
  /// Forward variance curve parameterization.
  pub xi0: RBergomiXi0,
}

impl RBergomiParams {
  pub fn project_in_place(&mut self) {
    self.hurst = self.hurst.clamp(H_MIN, H_MAX);
    self.rho = self.rho.clamp(-RHO_BOUND, RHO_BOUND);
    self.eta = self.eta.abs().max(ETA_MIN);
    self.xi0.project_in_place();
  }

  pub fn projected(mut self) -> Self {
    self.project_in_place();
    self
  }

  pub fn flattened_len(&self) -> usize {
    3 + self.xi0.flattened_len()
  }

  pub fn flatten(&self) -> Vec<f64> {
    let mut out = Vec::with_capacity(self.flattened_len());
    out.push(self.hurst);
    out.push(self.rho);
    out.push(self.eta);
    self.xi0.flatten_into(&mut out);
    out
  }

  pub fn assign_flattened(&mut self, values: &[f64]) {
    assert_eq!(
      values.len(),
      self.flattened_len(),
      "Flattened parameter vector length mismatch"
    );
    self.hurst = values[0];
    self.rho = values[1];
    self.eta = values[2];
    let mut offset = 3usize;
    self.xi0.assign_from_flattened(values, &mut offset);
  }

  fn validate(&self) -> Result<(), String> {
    if !self.hurst.is_finite() || self.hurst <= 0.0 || self.hurst >= 0.5 {
      return Err("RBergomiParams.hurst must be finite and in (0, 0.5)".to_string());
    }
    if !self.rho.is_finite() || self.rho.abs() > 1.0 {
      return Err("RBergomiParams.rho must be finite and in [-1, 1]".to_string());
    }
    if !self.eta.is_finite() || self.eta <= 0.0 {
      return Err("RBergomiParams.eta must be finite and positive".to_string());
    }
    self.xi0.validate()
  }
}

#[derive(Clone, Debug)]
pub struct RBergomiMarketSlice {
  /// Maturity $T_j$ in years.
  pub maturity: f64,
  /// Market terminal samples $\{S_{T_j}^{\mathrm{MKT},(m)}\}$.
  pub terminal_samples: Vec<f64>,
}

impl RBergomiMarketSlice {
  fn validate(&self) -> Result<(), String> {
    if !self.maturity.is_finite() || self.maturity <= 0.0 {
      return Err("RBergomiMarketSlice.maturity must be finite and positive".to_string());
    }
    if self.terminal_samples.is_empty() {
      return Err("RBergomiMarketSlice.terminal_samples cannot be empty".to_string());
    }
    if self.terminal_samples.iter().any(|x| !x.is_finite()) {
      return Err("RBergomiMarketSlice.terminal_samples must be finite".to_string());
    }
    Ok(())
  }
}

#[derive(Clone, Debug)]
pub struct RBergomiCalibrationConfig {
  /// Monte Carlo path count per maturity in each objective evaluation.
  pub paths: usize,
  /// Time discretization granularity, total steps = ceil(T * steps_per_year).
  pub steps_per_year: usize,
  /// Number of exponentials in mSOE kernel approximation.
  pub msoe_terms: usize,
  /// Max number of optimizer iterations.
  pub max_iters: usize,
  /// Base learning rate for projected Adam.
  pub learning_rate: f64,
  /// Relative finite-difference bump for numeric gradient.
  pub finite_diff_eps: f64,
  /// Adam $\beta_1$.
  pub adam_beta1: f64,
  /// Adam $\beta_2$.
  pub adam_beta2: f64,
  /// Adam numerical epsilon.
  pub adam_eps: f64,
  /// Common-random-number seed used in objective evaluations.
  pub random_seed: u64,
  /// Optional target tolerance (e.g. bid-ask derived) for early stop.
  pub stop_loss: Option<f64>,
  /// Stop if absolute loss improvement drops below this threshold.
  pub improvement_tol: f64,
}

impl Default for RBergomiCalibrationConfig {
  fn default() -> Self {
    Self {
      paths: 1_024,
      steps_per_year: 128,
      msoe_terms: 12,
      max_iters: 60,
      learning_rate: 0.05,
      finite_diff_eps: 1e-3,
      adam_beta1: 0.9,
      adam_beta2: 0.999,
      adam_eps: 1e-8,
      random_seed: 42,
      stop_loss: None,
      improvement_tol: 1e-6,
    }
  }
}

#[derive(Clone, Debug)]
pub struct RBergomiCalibrationHistory {
  pub iteration: usize,
  pub params: RBergomiParams,
  /// Pairs of `(maturity, W1)` for this iteration.
  pub maturity_losses: Vec<(f64, f64)>,
  /// Average Wasserstein loss across maturities.
  pub loss: f64,
}

#[derive(Clone, Debug)]
pub struct RBergomiCalibrationResult {
  pub initial_params: RBergomiParams,
  pub calibrated_params: RBergomiParams,
  pub initial_loss: f64,
  pub final_loss: f64,
  pub maturity_losses: Vec<(f64, f64)>,
  pub iterations: usize,
  pub converged: bool,
}

#[derive(Clone)]
pub struct RBergomiCalibrator {
  /// Spot level $S_0$.
  pub s0: f64,
  /// Risk-free rate $r$.
  pub r: f64,
  /// Current parameter guess.
  pub params: RBergomiParams,
  /// Distribution targets for each maturity.
  pub market_slices: Vec<RBergomiMarketSlice>,
  /// Calibration and simulation settings.
  pub config: RBergomiCalibrationConfig,
  /// If true, store per-iteration snapshots.
  pub record_history: bool,
  history: Rc<RefCell<Vec<RBergomiCalibrationHistory>>>,
}

impl RBergomiCalibrator {
  pub fn new(
    s0: f64,
    r: f64,
    params: RBergomiParams,
    mut market_slices: Vec<RBergomiMarketSlice>,
    config: RBergomiCalibrationConfig,
    record_history: bool,
  ) -> Self {
    assert!(s0.is_finite() && s0 > 0.0, "s0 must be finite and positive");
    assert!(r.is_finite(), "r must be finite");
    assert!(config.paths > 0, "config.paths must be > 0");
    assert!(
      config.steps_per_year > 0,
      "config.steps_per_year must be > 0"
    );
    assert!(config.msoe_terms > 0, "config.msoe_terms must be > 0");
    assert!(config.max_iters > 0, "config.max_iters must be > 0");
    assert!(
      config.learning_rate.is_finite() && config.learning_rate > 0.0,
      "config.learning_rate must be finite and > 0"
    );
    assert!(
      config.finite_diff_eps.is_finite() && config.finite_diff_eps > 0.0,
      "config.finite_diff_eps must be finite and > 0"
    );
    assert!(
      config.improvement_tol.is_finite() && config.improvement_tol >= 0.0,
      "config.improvement_tol must be finite and >= 0"
    );
    params.validate().expect("Invalid initial rBergomi params");
    for slice in &market_slices {
      slice.validate().expect("Invalid market slice");
    }
    market_slices.sort_by(|a, b| a.maturity.total_cmp(&b.maturity));

    Self {
      s0,
      r,
      params: params.projected(),
      market_slices,
      config,
      record_history,
      history: Rc::new(RefCell::new(Vec::new())),
    }
  }

  pub fn history(&self) -> Vec<RBergomiCalibrationHistory> {
    self.history.borrow().clone()
  }

  pub fn set_initial_guess(&mut self, params: RBergomiParams) {
    self.params = params.projected();
  }

  pub fn set_record_history(&mut self, record: bool) {
    self.record_history = record;
  }

  /// Returns `L(theta)` and the per-maturity `W1` contributions.
  pub fn loss(&self, params: &RBergomiParams) -> (f64, Vec<(f64, f64)>) {
    let p = params.clone().projected();
    let mut per_maturity = Vec::with_capacity(self.market_slices.len());

    for (idx, slice) in self.market_slices.iter().enumerate() {
      let seed = self.slice_seed(idx, slice.maturity);
      let model_samples = simulate_rbergomi_terminal_samples(
        &p,
        self.s0,
        self.r,
        slice.maturity,
        self.config.paths,
        self.config.steps_per_year,
        self.config.msoe_terms,
        seed,
      );
      let w1 = empirical_wasserstein_1(&model_samples, &slice.terminal_samples);
      per_maturity.push((slice.maturity, w1));
    }

    let avg = per_maturity.iter().map(|(_, w)| *w).sum::<f64>() / per_maturity.len().max(1) as f64;
    (avg, per_maturity)
  }

  fn finite_diff_gradient(&self, params: &RBergomiParams, theta: &[f64]) -> Vec<f64> {
    let dim = theta.len();
    let mut grad = vec![0.0_f64; dim];
    for i in 0..dim {
      let h = self.config.finite_diff_eps * theta[i].abs().max(1.0);

      let mut plus = theta.to_vec();
      plus[i] += h;
      let mut p_plus = params.clone();
      p_plus.assign_flattened(&plus);
      p_plus.project_in_place();
      let l_plus = self.loss(&p_plus).0;

      let mut minus = theta.to_vec();
      minus[i] -= h;
      let mut p_minus = params.clone();
      p_minus.assign_flattened(&minus);
      p_minus.project_in_place();
      let l_minus = self.loss(&p_minus).0;

      grad[i] = (l_plus - l_minus) / (2.0 * h);
    }
    grad
  }

  pub fn calibrate(&mut self) -> RBergomiCalibrationResult {
    self.history.borrow_mut().clear();

    self.params.project_in_place();
    let initial_params = self.params.clone();
    let mut current_params = self.params.clone();
    let mut theta = current_params.flatten();

    let (mut current_loss, mut current_per_maturity) = self.loss(&current_params);
    let initial_loss = current_loss;

    let mut best_params = current_params.clone();
    let mut best_loss = current_loss;
    let mut best_per_maturity = current_per_maturity.clone();

    if self.record_history {
      self.record_iteration(0, &current_params, current_loss, &current_per_maturity);
    }

    let mut m = vec![0.0; theta.len()];
    let mut v = vec![0.0; theta.len()];
    let mut b1_pow = 1.0_f64;
    let mut b2_pow = 1.0_f64;

    let mut iterations = 0usize;
    let mut converged = false;

    for iter in 1..=self.config.max_iters {
      iterations = iter;

      if let Some(stop) = self.config.stop_loss {
        if current_loss <= stop {
          converged = true;
          break;
        }
      }

      let grad = self.finite_diff_gradient(&current_params, &theta);

      b1_pow *= self.config.adam_beta1;
      b2_pow *= self.config.adam_beta2;

      let mut step = vec![0.0; theta.len()];
      for i in 0..theta.len() {
        m[i] = self.config.adam_beta1 * m[i] + (1.0 - self.config.adam_beta1) * grad[i];
        v[i] = self.config.adam_beta2 * v[i] + (1.0 - self.config.adam_beta2) * grad[i] * grad[i];
        let m_hat = m[i] / (1.0 - b1_pow);
        let v_hat = v[i] / (1.0 - b2_pow);
        step[i] = self.config.learning_rate * m_hat / (v_hat.sqrt() + self.config.adam_eps);
      }

      let mut accepted = false;
      let mut candidate_params = current_params.clone();
      let mut candidate_theta = theta.clone();
      let mut candidate_loss = current_loss;
      let mut candidate_per_maturity = current_per_maturity.clone();

      let mut step_scale = 1.0_f64;
      for _ in 0..8 {
        let mut proposal = theta.clone();
        for i in 0..proposal.len() {
          proposal[i] -= step_scale * step[i];
        }
        let mut p_try = current_params.clone();
        p_try.assign_flattened(&proposal);
        p_try.project_in_place();
        let projected = p_try.flatten();

        let (loss_try, per_maturity_try) = self.loss(&p_try);
        if loss_try.is_finite() && loss_try <= current_loss {
          accepted = true;
          candidate_params = p_try;
          candidate_theta = projected;
          candidate_loss = loss_try;
          candidate_per_maturity = per_maturity_try;
          break;
        }
        step_scale *= 0.5;
      }

      if !accepted {
        let grad_norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt().max(1e-12);
        let mut proposal = theta.clone();
        for i in 0..proposal.len() {
          proposal[i] -= self.config.learning_rate * grad[i] / grad_norm;
        }
        let mut p_try = current_params.clone();
        p_try.assign_flattened(&proposal);
        p_try.project_in_place();
        let projected = p_try.flatten();

        let (loss_try, per_maturity_try) = self.loss(&p_try);
        if loss_try.is_finite() && loss_try < current_loss {
          accepted = true;
          candidate_params = p_try;
          candidate_theta = projected;
          candidate_loss = loss_try;
          candidate_per_maturity = per_maturity_try;
        }
      }

      if !accepted {
        break;
      }

      let improvement = (current_loss - candidate_loss).abs();
      current_params = candidate_params;
      theta = candidate_theta;
      current_loss = candidate_loss;
      current_per_maturity = candidate_per_maturity;

      if current_loss < best_loss {
        best_loss = current_loss;
        best_params = current_params.clone();
        best_per_maturity = current_per_maturity.clone();
      }

      if self.record_history {
        self.record_iteration(iter, &current_params, current_loss, &current_per_maturity);
      }

      if improvement < self.config.improvement_tol {
        break;
      }
    }

    self.params = best_params.clone();
    if let Some(stop) = self.config.stop_loss {
      converged = converged || best_loss <= stop;
    }

    RBergomiCalibrationResult {
      initial_params,
      calibrated_params: best_params,
      initial_loss,
      final_loss: best_loss,
      maturity_losses: best_per_maturity,
      iterations,
      converged,
    }
  }

  fn record_iteration(
    &self,
    iteration: usize,
    params: &RBergomiParams,
    loss: f64,
    maturity_losses: &[(f64, f64)],
  ) {
    self.history.borrow_mut().push(RBergomiCalibrationHistory {
      iteration,
      params: params.clone(),
      maturity_losses: maturity_losses.to_vec(),
      loss,
    });
  }

  fn slice_seed(&self, idx: usize, maturity: f64) -> u64 {
    let a = 0x9E37_79B9_7F4A_7C15_u64;
    let b = 0xBF58_476D_1CE4_E5B9_u64;
    self
      .config
      .random_seed
      .wrapping_add(a.wrapping_mul((idx as u64).wrapping_add(1)))
      ^ maturity.to_bits().wrapping_mul(b)
  }
}

#[derive(Clone)]
struct MsoeEngine {
  h: f64,
  dt: f64,
  lambdas: Vec<f64>,
  weights: Vec<f64>,
  decay: Vec<f64>,
  second_moment: Vec<f64>,
  chol_l: Vec<Vec<f64>>,
}

impl MsoeEngine {
  fn new(h: f64, dt: f64, maturity: f64, steps: usize, terms: usize) -> Self {
    let (lambdas, weights) = build_msoe_kernel(h, dt, maturity, terms.max(2));
    let decay = lambdas.iter().map(|x| (-x * dt).exp()).collect::<Vec<_>>();
    let cov = build_step_covariance(h, dt, &lambdas);
    let l = cholesky_lower_with_jitter(cov);
    let mut chol_l = Vec::with_capacity(l.nrows());
    for row in 0..l.nrows() {
      let mut v = vec![0.0_f64; row + 1];
      for col in 0..=row {
        v[col] = l[(row, col)];
      }
      chol_l.push(v);
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

  fn dim(&self) -> usize {
    self.lambdas.len() + 2
  }

  fn terms(&self) -> usize {
    self.lambdas.len()
  }

  fn transform(&self, z: &[f64], out: &mut [f64]) {
    debug_assert_eq!(z.len(), self.dim());
    debug_assert_eq!(out.len(), self.dim());
    for row in 0..self.dim() {
      let mut acc = 0.0;
      for (col, l_rc) in self.chol_l[row].iter().enumerate() {
        acc += l_rc * z[col];
      }
      out[row] = acc;
    }
  }
}

/// Empirical $W_1$ distance between two 1D sample sets using quantile coupling.
pub fn empirical_wasserstein_1(x: &[f64], y: &[f64]) -> f64 {
  let mut xs = x
    .iter()
    .copied()
    .filter(|v| v.is_finite())
    .collect::<Vec<f64>>();
  let mut ys = y
    .iter()
    .copied()
    .filter(|v| v.is_finite())
    .collect::<Vec<f64>>();

  if xs.is_empty() || ys.is_empty() {
    return f64::INFINITY;
  }

  xs.sort_by(|a, b| a.total_cmp(b));
  ys.sort_by(|a, b| a.total_cmp(b));

  let m = xs.len().max(ys.len());
  let mut acc = 0.0;
  for i in 0..m {
    let u = (i as f64 + 0.5) / m as f64;
    let qx = quantile_sorted(&xs, u);
    let qy = quantile_sorted(&ys, u);
    acc += (qx - qy).abs();
  }
  acc / m as f64
}

/// Computes bid-ask calibration tolerance:
/// $\varepsilon = \frac1M \sum_j |\mathrm{ask}_j - \mathrm{bid}_j|$.
pub fn bid_ask_tolerance(bid: &[f64], ask: &[f64]) -> f64 {
  if bid.is_empty() || ask.is_empty() {
    return 0.0;
  }
  let m = bid.len().min(ask.len());
  bid
    .iter()
    .zip(ask.iter())
    .take(m)
    .map(|(b, a)| (a - b).abs())
    .sum::<f64>()
    / m as f64
}

/// Simulates terminal prices under the rBergomi model with an mSOE approximation
/// for the Volterra kernel history term.
pub fn simulate_rbergomi_terminal_samples(
  params: &RBergomiParams,
  s0: f64,
  r: f64,
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
      let mut rng = StdRng::seed_from_u64(
        seed
          .wrapping_add(0xD134_2543_DE82_EF95_u64.wrapping_mul((path_idx as u64).wrapping_add(1))),
      );
      let dim = engine.dim();
      let mut z = vec![0.0_f64; dim];
      let mut xi = vec![0.0_f64; dim];
      let mut history = vec![0.0_f64; engine.terms()];

      let mut s = s0.max(1e-12);
      let mut v_prev = params.xi0.value(0.0).max(XI0_MIN);

      for step in 1..=steps {
        for zi in z.iter_mut() {
          *zi = StandardNormal.sample(&mut rng);
        }
        engine.transform(&z, &mut xi);

        let d_w = xi[0];
        let d_w_perp: f64 = StandardNormal.sample(&mut rng);
        let d_w_perp = d_w_perp * sqrt_dt;

        let drift = (r - 0.5 * v_prev) * dt;
        let diffusion = v_prev.sqrt() * (rho * d_w + rho_orth * d_w_perp);
        s *= (drift + diffusion).exp();

        let mut past_sum = 0.0;
        for k in 0..engine.terms() {
          past_sum += engine.weights[k] * history[k];
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

fn quantile_sorted(sorted: &[f64], u: f64) -> f64 {
  if sorted.len() == 1 {
    return sorted[0];
  }
  let z = u.clamp(0.0, 1.0);
  let pos = z * (sorted.len() as f64 - 1.0);
  let lo = pos.floor() as usize;
  let hi = pos.ceil() as usize;
  if lo == hi {
    sorted[lo]
  } else {
    let w = pos - lo as f64;
    sorted[lo] * (1.0 - w) + sorted[hi] * w
  }
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

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_empirical_wasserstein_1_matches_simple_case() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![2.0, 3.0, 4.0];
    let w1 = empirical_wasserstein_1(&a, &b);
    assert!((w1 - 1.0).abs() < 1e-12);
  }

  #[test]
  fn test_rbergomi_calibration_reduces_loss_on_synthetic_data() {
    let true_params = RBergomiParams {
      hurst: 0.12,
      rho: -0.72,
      eta: 1.6,
      xi0: RBergomiXi0::Constant(0.04),
    };

    let maturities = vec![0.25, 0.5, 1.0];
    let mut market_slices = Vec::with_capacity(maturities.len());
    for (i, &t) in maturities.iter().enumerate() {
      let market_samples = simulate_rbergomi_terminal_samples(
        &true_params,
        100.0,
        0.01,
        t,
        512,
        96,
        12,
        7_777 + i as u64,
      );
      market_slices.push(RBergomiMarketSlice {
        maturity: t,
        terminal_samples: market_samples,
      });
    }

    let init_params = RBergomiParams {
      hurst: 0.30,
      rho: -0.10,
      eta: 0.80,
      xi0: RBergomiXi0::Constant(0.02),
    };

    let config = RBergomiCalibrationConfig {
      paths: 512,
      steps_per_year: 96,
      msoe_terms: 12,
      max_iters: 12,
      learning_rate: 0.08,
      finite_diff_eps: 5e-3,
      adam_beta1: 0.9,
      adam_beta2: 0.99,
      adam_eps: 1e-8,
      random_seed: 123_456,
      stop_loss: None,
      improvement_tol: 1e-5,
    };

    let mut calibrator = RBergomiCalibrator::new(
      100.0,
      0.01,
      init_params.clone(),
      market_slices,
      config,
      true,
    );

    let result = calibrator.calibrate();

    println!(
      "rBergomi calibration: initial_loss={:.6}, final_loss={:.6}, iterations={}",
      result.initial_loss, result.final_loss, result.iterations
    );
    println!("initial params: {:?}", result.initial_params);
    println!("estimated params: {:?}", result.calibrated_params);
    println!("per-maturity W1: {:?}", result.maturity_losses);

    assert!(result.final_loss <= result.initial_loss);
    assert!(result.calibrated_params.hurst > 0.0 && result.calibrated_params.hurst < 0.5);
    assert!(result.calibrated_params.rho.abs() < 1.0);
    assert!(result.calibrated_params.eta > 0.0);

    if let RBergomiXi0::Constant(xi0_hat) = result.calibrated_params.xi0 {
      assert!(xi0_hat > 0.0);
    } else {
      panic!("Expected constant xi0 in this test");
    }
  }
}
