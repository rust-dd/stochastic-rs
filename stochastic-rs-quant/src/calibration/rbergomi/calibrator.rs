use std::collections::HashMap;

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use super::loss::empirical_wasserstein_1;
use super::params::RBergomiCalibrationConfig;
use super::params::RBergomiMarketSlice;
use super::params::RBergomiParams;
use super::simulation::simulate_rbergomi_terminal_samples;
use crate::CalibrationLossScore;
use crate::LossMetric;
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationHistory;
use crate::calibration::CalibrationOptions;
use crate::calibration::Knob;
use crate::calibration::quote::assert_spot_and_rates;
use crate::pricing::rbergomi::RBergomiPricer;
use crate::traits::ToModel;

impl<D> ToModel for Calibrated<RBergomiParams, D> {
  type Model = RBergomiPricer;

  fn to_model(&self) -> RBergomiPricer {
    RBergomiPricer::new(self.params.clone())
  }
}

/// rBergomi fit to terminal-price samples by projected Adam on the mean Wasserstein-1 distance.
#[derive(Clone, Debug)]
pub struct RBergomiCalibrator<S: SeedExt = Unseeded> {
  s: f64,
  r: f64,
  /// Continuous dividend yield (or foreign rate for FX).
  q: f64,
  /// Projected onto the parameter box when a fit starts.
  pub start: RBergomiParams,
  /// Sorted by maturity in `new`; `calibrate` reports a slice by its index in this order.
  pub market_slices: Vec<RBergomiMarketSlice>,
  config: RBergomiCalibrationConfig,
  /// Adam reads `max_iterations` (default 60) and `record_history`; its stop is `config`'s, so
  /// `tolerance`, `patience`, `loss_metrics` and `regularization` are `Unsupported`.
  pub options: CalibrationOptions,
  /// A fit reads one base from a clone, every loss sharing it as its common random numbers.
  pub seed: S,
}

impl RBergomiCalibrator {
  /// Panics unless `s` is finite and positive, `r` and `q` finite and `config`'s counts and rates
  /// in range; the start and the slices are data, which `calibrate` checks.
  pub fn new(
    s: f64,
    r: f64,
    q: f64,
    start: RBergomiParams,
    mut market_slices: Vec<RBergomiMarketSlice>,
    config: RBergomiCalibrationConfig,
  ) -> Self {
    assert_spot_and_rates(s, r, q);
    for (name, count) in [
      ("config.paths", config.paths),
      ("config.steps_per_year", config.steps_per_year),
      ("config.msoe_terms", config.msoe_terms),
    ] {
      assert!(
        count > 0,
        "{name} must satisfy `{name} > 0`, got {name} = {count}"
      );
    }
    let rate = config.learning_rate;
    assert!(
      rate.is_finite() && rate > 0.0,
      "config.learning_rate must satisfy `config.learning_rate.is_finite() && config.learning_rate > 0.0`, got config.learning_rate = {rate:?}"
    );
    let eps = config.finite_diff_eps;
    assert!(
      eps.is_finite() && eps > 0.0,
      "config.finite_diff_eps must satisfy `config.finite_diff_eps.is_finite() && config.finite_diff_eps > 0.0`, got config.finite_diff_eps = {eps:?}"
    );
    let tol = config.improvement_tol;
    assert!(
      tol.is_finite() && tol >= 0.0,
      "config.improvement_tol must satisfy `config.improvement_tol.is_finite() && config.improvement_tol >= 0.0`, got config.improvement_tol = {tol:?}"
    );
    market_slices.sort_by(|a, b| a.tau.total_cmp(&b.tau));
    Self {
      s,
      r,
      q,
      start,
      market_slices,
      config,
      options: CalibrationOptions::default(),
      seed: Unseeded,
    }
  }
}

impl<S: SeedExt> RBergomiCalibrator<S> {
  pub fn with_seed<S2: SeedExt>(self, seed: S2) -> RBergomiCalibrator<S2> {
    RBergomiCalibrator {
      s: self.s,
      r: self.r,
      q: self.q,
      start: self.start,
      market_slices: self.market_slices,
      config: self.config,
      options: self.options,
      seed,
    }
  }

  pub fn s(&self) -> f64 {
    self.s
  }

  pub fn r(&self) -> f64 {
    self.r
  }

  pub fn q(&self) -> f64 {
    self.q
  }

  pub fn with_options(mut self, options: CalibrationOptions) -> Self {
    self.options = options;
    self
  }

  /// Returns `L(theta)` and the per-maturity `W1` contributions, at a base the seed draws.
  pub fn loss(&self, params: &RBergomiParams) -> (f64, Vec<(f64, f64)>) {
    self.loss_at(params, self.seed.clone().next_seed())
  }

  fn loss_at(&self, params: &RBergomiParams, base: u64) -> (f64, Vec<(f64, f64)>) {
    let p = params.clone().projected();
    let mut per_maturity = Vec::with_capacity(self.market_slices.len());

    for (idx, slice) in self.market_slices.iter().enumerate() {
      let seed = Deterministic::new(slice_seed(base, idx, slice.tau));
      let model_samples = simulate_rbergomi_terminal_samples(
        &p,
        self.s,
        self.r,
        self.q,
        slice.tau,
        self.config.paths,
        self.config.steps_per_year,
        self.config.msoe_terms,
        &seed,
      );
      let w1 = empirical_wasserstein_1(&model_samples, &slice.terminal_samples);
      per_maturity.push((slice.tau, w1));
    }

    let avg = per_maturity.iter().map(|(_, w)| *w).sum::<f64>() / per_maturity.len().max(1) as f64;
    (avg, per_maturity)
  }

  fn finite_diff_gradient(&self, params: &RBergomiParams, theta: &[f64], base: u64) -> Vec<f64> {
    let dim = theta.len();
    let mut grad = vec![0.0_f64; dim];
    for i in 0..dim {
      let h = self.config.finite_diff_eps * theta[i].abs().max(1.0);

      let mut plus = theta.to_vec();
      plus[i] += h;
      let mut p_plus = params.clone();
      p_plus.assign_flattened(&plus);
      p_plus.project_in_place();
      let l_plus = self.loss_at(&p_plus, base).0;

      let mut minus = theta.to_vec();
      minus[i] -= h;
      let mut p_minus = params.clone();
      p_minus.assign_flattened(&minus);
      p_minus.project_in_place();
      let l_minus = self.loss_at(&p_minus, base).0;

      grad[i] = (l_plus - l_minus) / (2.0 * h);
    }
    grad
  }

  fn solve(&self, start: RBergomiParams) -> Calibrated<RBergomiParams> {
    let config = &self.config;
    let base = self.seed.clone().next_seed();
    let mut history = Vec::new();
    let mut record = |iteration, params: &RBergomiParams, loss, per_maturity: &[(f64, f64)]| {
      if self.options.record_history {
        let w1 = per_maturity
          .iter()
          .map(|&(_, w)| w)
          .collect::<Array1<f64>>();
        let entry = CalibrationHistory::new(iteration, params.clone(), w1, wasserstein_loss(loss));
        history.push(entry);
      }
    };

    let mut current_params = start.projected();
    let mut theta = current_params.flatten();
    let (mut current_loss, mut current_per_maturity) = self.loss_at(&current_params, base);
    let mut best_params = current_params.clone();
    let mut best_loss = current_loss;
    record(0, &current_params, current_loss, &current_per_maturity);

    let mut m = vec![0.0; theta.len()];
    let mut v = vec![0.0; theta.len()];
    let mut b1_pow = 1.0_f64;
    let mut b2_pow = 1.0_f64;

    let mut iterations = 0usize;
    let mut exit = Exit::Cap;

    for iter in 1..=self.options.max_iterations.unwrap_or(60) {
      iterations = iter;

      if let Some(stop_loss) = config.stop_loss
        && current_loss <= stop_loss
      {
        exit = Exit::StopLoss;
        break;
      }

      let grad = self.finite_diff_gradient(&current_params, &theta, base);

      b1_pow *= config.adam_beta1;
      b2_pow *= config.adam_beta2;

      let mut step = vec![0.0; theta.len()];
      for i in 0..theta.len() {
        m[i] = config.adam_beta1 * m[i] + (1.0 - config.adam_beta1) * grad[i];
        v[i] = config.adam_beta2 * v[i] + (1.0 - config.adam_beta2) * grad[i] * grad[i];
        let m_hat = m[i] / (1.0 - b1_pow);
        let v_hat = v[i] / (1.0 - b2_pow);
        step[i] = config.learning_rate * m_hat / (v_hat.sqrt() + config.adam_eps);
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

        let (loss_try, per_maturity_try) = self.loss_at(&p_try, base);
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
          proposal[i] -= config.learning_rate * grad[i] / grad_norm;
        }
        let mut p_try = current_params.clone();
        p_try.assign_flattened(&proposal);
        p_try.project_in_place();
        let projected = p_try.flatten();

        let (loss_try, per_maturity_try) = self.loss_at(&p_try, base);
        if loss_try.is_finite() && loss_try < current_loss {
          accepted = true;
          candidate_params = p_try;
          candidate_theta = projected;
          candidate_loss = loss_try;
          candidate_per_maturity = per_maturity_try;
        }
      }

      if !accepted {
        exit = Exit::NoDescent;
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
      }

      record(iter, &current_params, current_loss, &current_per_maturity);

      if improvement < config.improvement_tol {
        exit = Exit::Improvement;
        break;
      }
    }

    // The cap's last step can bring the loss to the stop the next iteration would test.
    if exit == Exit::Cap
      && config
        .stop_loss
        .is_some_and(|stop_loss| best_loss <= stop_loss)
    {
      exit = Exit::StopLoss;
    }
    let converged = matches!(exit, Exit::StopLoss | Exit::Improvement);
    Calibrated::new(best_params, wasserstein_loss(best_loss), converged)
      .with_iterations(iterations)
      .with_message(exit.message())
      .with_history(history)
  }
}

fn slice_seed(base: u64, idx: usize, tau: f64) -> u64 {
  let a = 0x9E37_79B9_7F4A_7C15_u64;
  let b = 0xBF58_476D_1CE4_E5B9_u64;
  base.wrapping_add(a.wrapping_mul((idx as u64).wrapping_add(1))) ^ tau.to_bits().wrapping_mul(b)
}

/// Why the Adam loop ended; the fit converged only on `config`'s own criterion.
#[derive(Clone, Copy, PartialEq)]
enum Exit {
  StopLoss,
  Improvement,
  NoDescent,
  Cap,
}

impl Exit {
  fn message(self) -> &'static str {
    match self {
      Exit::StopLoss => "loss reached stop_loss",
      Exit::Improvement => "improvement below improvement_tol",
      Exit::NoDescent => "no descent step",
      Exit::Cap => "iteration cap reached",
    }
  }
}

/// The mean `W1` as `Mse` and its square root as `Rmse`, on a price fit's scale.
fn wasserstein_loss(w1: f64) -> CalibrationLossScore {
  let scores = HashMap::from([(LossMetric::Mse, w1), (LossMetric::Rmse, w1.abs().sqrt())]);
  CalibrationLossScore { scores }
}

impl<S: SeedExt> crate::traits::Calibrator for RBergomiCalibrator<S> {
  type Params = RBergomiParams;
  type Output = Calibrated<RBergomiParams>;
  type Error = QuantError;

  /// A start replaces `self.start`; `InvalidInput` unless it and each slice lie in their domain,
  /// `InsufficientData` for no slice, and the `CalibrationOptions` errors.
  fn calibrate(
    &self,
    initial: Option<RBergomiParams>,
  ) -> Result<Calibrated<RBergomiParams>, QuantError> {
    let unread = [
      Knob::Tolerance,
      Knob::Patience,
      Knob::LossMetrics,
      Knob::Regularization,
    ];
    self.options.check(&unread)?;
    if self.market_slices.is_empty() {
      return Err(QuantError::InsufficientData {
        what: "market slices",
        needed: 1,
        got: 0,
      });
    }
    for (index, slice) in self.market_slices.iter().enumerate() {
      slice.validate(index)?;
    }
    let start = initial.unwrap_or_else(|| self.start.clone());
    start.validate()?;
    Ok(self.solve(start))
  }
}
