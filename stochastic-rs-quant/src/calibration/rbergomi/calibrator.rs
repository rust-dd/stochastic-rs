use std::cell::RefCell;
use std::rc::Rc;

use super::loss::empirical_wasserstein_1;
use super::params::{
  RBergomiCalibrationConfig, RBergomiCalibrationHistory, RBergomiMarketSlice, RBergomiParams,
};
use super::result::RBergomiCalibrationResult;
use super::simulation::simulate_rbergomi_terminal_samples;

#[derive(Clone)]
pub struct RBergomiCalibrator {
  /// Spot level $S_0$.
  pub s0: f64,
  /// Risk-free rate $r$.
  pub r: f64,
  /// Continuous dividend yield $q$ (or foreign rate for FX). Defaults to 0
  /// when constructed via [`Self::new`]; set with [`Self::with_dividend_yield`].
  pub q: f64,
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
  /// Construct a calibrator. Returns an error if any input is invalid
  /// (non-finite spot/rate, non-positive config counts, malformed params,
  /// or malformed market slices).
  pub fn new(
    s0: f64,
    r: f64,
    params: RBergomiParams,
    mut market_slices: Vec<RBergomiMarketSlice>,
    config: RBergomiCalibrationConfig,
    record_history: bool,
  ) -> anyhow::Result<Self> {
    if !(s0.is_finite() && s0 > 0.0) {
      anyhow::bail!("s0 must be finite and positive, got {s0}");
    }
    if !r.is_finite() {
      anyhow::bail!("r must be finite, got {r}");
    }
    if config.paths == 0 {
      anyhow::bail!("config.paths must be > 0");
    }
    if config.steps_per_year == 0 {
      anyhow::bail!("config.steps_per_year must be > 0");
    }
    if config.msoe_terms == 0 {
      anyhow::bail!("config.msoe_terms must be > 0");
    }
    if config.max_iters == 0 {
      anyhow::bail!("config.max_iters must be > 0");
    }
    if !(config.learning_rate.is_finite() && config.learning_rate > 0.0) {
      anyhow::bail!(
        "config.learning_rate must be finite and > 0, got {}",
        config.learning_rate
      );
    }
    if !(config.finite_diff_eps.is_finite() && config.finite_diff_eps > 0.0) {
      anyhow::bail!(
        "config.finite_diff_eps must be finite and > 0, got {}",
        config.finite_diff_eps
      );
    }
    if !(config.improvement_tol.is_finite() && config.improvement_tol >= 0.0) {
      anyhow::bail!(
        "config.improvement_tol must be finite and >= 0, got {}",
        config.improvement_tol
      );
    }
    params
      .validate()
      .map_err(|e| anyhow::anyhow!("invalid initial rBergomi params: {e}"))?;
    for (idx, slice) in market_slices.iter().enumerate() {
      slice
        .validate()
        .map_err(|e| anyhow::anyhow!("invalid market slice at index {idx}: {e}"))?;
    }
    market_slices.sort_by(|a, b| a.maturity.total_cmp(&b.maturity));

    Ok(Self {
      s0,
      r,
      q: 0.0,
      params: params.projected(),
      market_slices,
      config,
      record_history,
      history: Rc::new(RefCell::new(Vec::new())),
    })
  }

  /// Set the continuous dividend yield $q$ (or foreign rate for FX). Default 0.
  pub fn with_dividend_yield(mut self, q: f64) -> Self {
    assert!(q.is_finite(), "q must be finite");
    self.q = q;
    self
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
        self.q,
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

  fn solve(&mut self) -> RBergomiCalibrationResult {
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

      if let Some(stop) = self.config.stop_loss
        && current_loss <= stop
      {
        converged = true;
        break;
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

impl crate::traits::Calibrator for RBergomiCalibrator {
  type InitialGuess = RBergomiParams;
  type Params = RBergomiParams;
  type Output = RBergomiCalibrationResult;
  type Error = anyhow::Error;

  fn calibrate(&self, initial: Option<Self::InitialGuess>) -> Result<Self::Output, Self::Error> {
    let mut this = self.clone();
    if let Some(p) = initial {
      this.params = p;
    }
    Ok(this.solve())
  }
}
