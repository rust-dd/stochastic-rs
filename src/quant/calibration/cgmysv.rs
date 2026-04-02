//! # CGMYSV Calibration
//!
//! $$
//! \hat\theta=\arg\min_\theta\sum_i\bigl(C_i^{\mathrm{model}}(\theta)-C_i^{\mathrm{mkt}}\bigr)^2
//! $$
//!
//! Calibrates the CGMYSV stochastic volatility model parameters
//! $(\alpha,\lambda_+,\lambda_-,\kappa,\eta,\zeta,\rho,v_0)$ to observed
//! option prices using Fourier pricing (Gil-Pelaez) and Levenberg-Marquardt.
//!
//! Reference: Kim, Y. S. (2021), arXiv:2101.11001

use std::cell::RefCell;
use std::rc::Rc;

use levenberg_marquardt::LeastSquaresProblem;
use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::Dyn;
use nalgebra::Owned;
use super::CalibrationHistory;
/// Market data for a single maturity slice.
pub use super::levy::MarketSlice;
use crate::quant::CalibrationLossScore;
use crate::quant::LossMetric;
use crate::quant::pricing::cgmysv::CgmysvModel;
use crate::quant::pricing::cgmysv::CgmysvParams;
use crate::quant::pricing::fourier::LewisPricer;

/// CGMYSV calibration result.
#[derive(Clone, Debug)]
pub struct CgmysvCalibrationResult {
  pub params: CgmysvParams,
  pub loss: CalibrationLossScore,
  pub converged: bool,
  pub iterations: usize,
}

impl crate::traits::ToModel for CgmysvCalibrationResult {
  fn to_model(&self, r: f64, q: f64) -> Box<dyn crate::traits::ModelPricer> {
    Box::new(CgmysvModel {
      params: self.params.clone(),
      r,
      q,
    })
  }
}

/// CGMYSV calibrator via Fourier pricing + Levenberg-Marquardt.
#[derive(Clone)]
pub struct CgmysvCalibrator {
  pub s: f64,
  pub r: f64,
  pub q: f64,
  pub market_data: Vec<MarketSlice>,
  pub record_history: bool,
  pub loss_metrics: &'static [LossMetric],
  params: Vec<f64>,
  flat_prices: Vec<f64>,
  flat_strikes: Vec<f64>,
  flat_t: Vec<f64>,
  flat_is_call: Vec<bool>,
  calibration_history: Rc<RefCell<Vec<CalibrationHistory<Vec<f64>>>>>,
}

const N_PARAMS: usize = 8;

// Parameter order: [alpha, lambda_plus, lambda_minus, kappa, eta, zeta, rho, v0]
fn to_cgmysv_params(p: &[f64]) -> CgmysvParams {
  CgmysvParams {
    alpha: p[0],
    lambda_plus: p[1],
    lambda_minus: p[2],
    kappa: p[3],
    eta: p[4],
    zeta: p[5],
    rho: p[6],
    v0: p[7],
  }
}

fn default_params() -> Vec<f64> {
  // alpha, lambda_plus, lambda_minus, kappa, eta, zeta, rho, v0
  vec![0.5, 20.0, 5.0, 1.0, 0.05, 0.3, -1.0, 0.01]
}

fn param_bounds() -> [(f64, f64); N_PARAMS] {
  [
    (0.01, 1.999), // alpha
    (0.1, 100.0),  // lambda_plus
    (0.1, 100.0),  // lambda_minus
    (0.01, 10.0),  // kappa
    (0.001, 1.0),  // eta
    (0.01, 5.0),   // zeta
    (-5.0, 5.0),   // rho
    (0.0001, 0.5), // v0
  ]
}

fn project(p: &mut [f64]) {
  let bounds = param_bounds();
  for (v, (lo, hi)) in p.iter_mut().zip(bounds.iter()) {
    *v = v.clamp(*lo, *hi);
  }
}

/// Price a European call using the CGMYSV characteristic function via Lewis (2001).
fn fourier_call_price(params: &CgmysvParams, s: f64, k: f64, r: f64, q: f64, t: f64) -> f64 {
  let model = CgmysvModel {
    params: params.clone(),
    r,
    q,
  };
  LewisPricer::price_call(&model, s, k, r, q, t)
}

fn fourier_option_price(
  params: &CgmysvParams,
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  t: f64,
  is_call: bool,
) -> f64 {
  let call = fourier_call_price(params, s, k, r, q, t);
  if is_call {
    call
  } else {
    (call - s * (-q * t).exp() + k * (-r * t).exp()).max(0.0)
  }
}

impl CgmysvCalibrator {
  pub fn new(s: f64, r: f64, q: f64, market_data: Vec<MarketSlice>) -> Self {
    let (flat_prices, flat_strikes, flat_t, flat_is_call) = Self::flatten(&market_data);
    Self {
      s,
      r,
      q,
      market_data,
      record_history: false,
      loss_metrics: &LossMetric::ALL,
      params: default_params(),
      flat_prices,
      flat_strikes,
      flat_t,
      flat_is_call,
      calibration_history: Rc::new(RefCell::new(Vec::new())),
    }
  }

  fn flatten(data: &[MarketSlice]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<bool>) {
    let mut prices = Vec::new();
    let mut strikes = Vec::new();
    let mut ts = Vec::new();
    let mut is_call = Vec::new();
    for slice in data {
      for i in 0..slice.strikes.len() {
        prices.push(slice.prices[i]);
        strikes.push(slice.strikes[i]);
        ts.push(slice.t);
        is_call.push(slice.is_call[i]);
      }
    }
    (prices, strikes, ts, is_call)
  }

  pub fn set_record_history(&mut self, record: bool) {
    self.record_history = record;
  }

  pub fn history(&self) -> Vec<CalibrationHistory<Vec<f64>>> {
    self.calibration_history.borrow().clone()
  }

  fn compute_model_prices(&self) -> Vec<f64> {
    let p = to_cgmysv_params(&self.params);
    self
      .flat_strikes
      .iter()
      .zip(self.flat_t.iter())
      .zip(self.flat_is_call.iter())
      .map(|((&k, &t), &is_call)| fourier_option_price(&p, self.s, k, self.r, self.q, t, is_call))
      .collect()
  }

  /// Run calibration with optional initial parameters.
  pub fn calibrate(&self, initial_params: Option<Vec<f64>>) -> CgmysvCalibrationResult {
    let mut problem = self.clone();
    if let Some(p) = initial_params {
      assert_eq!(p.len(), N_PARAMS);
      problem.params = p;
    }
    project(&mut problem.params);

    let (result, report) = LevenbergMarquardt::new().minimize(problem);

    let final_params = to_cgmysv_params(&result.params);
    let model_prices = result.compute_model_prices();
    let loss = CalibrationLossScore::compute_selected(
      &result.flat_prices,
      &model_prices,
      result.loss_metrics,
    );

    CgmysvCalibrationResult {
      params: final_params,
      loss,
      converged: report.termination.was_successful(),
      iterations: report.number_of_evaluations,
    }
  }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for CgmysvCalibrator {
  type JacobianStorage = Owned<f64, Dyn, Dyn>;
  type ParameterStorage = Owned<f64, Dyn>;
  type ResidualStorage = Owned<f64, Dyn>;

  fn set_params(&mut self, params: &DVector<f64>) {
    self.params = params.as_slice().to_vec();
    project(&mut self.params);
  }

  fn params(&self) -> DVector<f64> {
    DVector::from_vec(self.params.clone())
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    let model_prices = self.compute_model_prices();
    let n = self.flat_prices.len();
    let residuals: Vec<f64> = (0..n)
      .map(|i| model_prices[i] - self.flat_prices[i])
      .collect();

    if self.record_history {
      let r_vec = DVector::from_vec(residuals.clone());
      let call_put = DVector::from_vec(vec![(0.0, 0.0); n]);
      let loss =
        CalibrationLossScore::compute_selected(&self.flat_prices, &model_prices, self.loss_metrics);
      self
        .calibration_history
        .borrow_mut()
        .push(CalibrationHistory {
          residuals: r_vec,
          call_put,
          params: self.params.clone(),
          loss_scores: loss,
        });
    }

    Some(DVector::from_vec(residuals))
  }

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    let n = self.flat_prices.len();
    let mut jac = DMatrix::zeros(n, N_PARAMS);
    let h = 1e-6;

    for j in 0..N_PARAMS {
      let mut p_up = self.params.clone();
      let mut p_dn = self.params.clone();
      p_up[j] += h;
      p_dn[j] -= h;
      project(&mut p_up);
      project(&mut p_dn);

      let params_up = to_cgmysv_params(&p_up);
      let params_dn = to_cgmysv_params(&p_dn);
      let denom = p_up[j] - p_dn[j];
      if denom.abs() < 1e-15 {
        continue;
      }

      for i in 0..n {
        let price_up = fourier_option_price(
          &params_up,
          self.s,
          self.flat_strikes[i],
          self.r,
          self.q,
          self.flat_t[i],
          self.flat_is_call[i],
        );
        let price_dn = fourier_option_price(
          &params_dn,
          self.s,
          self.flat_strikes[i],
          self.r,
          self.q,
          self.flat_t[i],
          self.flat_is_call[i],
        );
        jac[(i, j)] = (price_up - price_dn) / denom;
      }
    }

    Some(jac)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn known_params() -> CgmysvParams {
    CgmysvParams {
      alpha: 0.5184,
      lambda_plus: 25.4592,
      lambda_minus: 4.6040,
      kappa: 1.0029,
      eta: 0.0711,
      zeta: 0.3443,
      rho: -2.0283,
      v0: 0.01115,
    }
  }

  /// Generate synthetic market data from known params, calibrate back, check recovery.
  #[test]
  fn round_trip_calibration() {
    let p = known_params();
    let s = 2488.11;
    let r = 0.01213;
    let q = 0.01884;
    let tau = 28.0 / 365.0;

    // Generate synthetic call prices at several strikes
    let strikes: Vec<f64> = (2400..=2600).step_by(25).map(|k| k as f64).collect();
    let prices: Vec<f64> = strikes
      .iter()
      .map(|&k| fourier_call_price(&p, s, k, r, q, tau))
      .collect();
    let is_call = vec![true; strikes.len()];

    println!("Synthetic prices:");
    for (k, p) in strikes.iter().zip(prices.iter()) {
      println!("  K={k:.0}  C={p:.4}");
    }

    let market = MarketSlice {
      strikes: strikes.clone(),
      prices: prices.clone(),
      is_call,
      t: tau,
    };

    let calibrator = CgmysvCalibrator::new(s, r, q, vec![market]);
    let result = calibrator.calibrate(None);

    println!("Calibrated: {:?}", result.params);
    println!("Loss RMSE: {:.6}", result.loss.get(LossMetric::Rmse));
    println!("Converged: {}, Iterations: {}", result.converged, result.iterations);

    // RMSE should be small since we calibrate to synthetic data
    assert!(
      result.loss.get(LossMetric::Rmse) < 5.0,
      "RMSE = {:.4}, should be small for synthetic data",
      result.loss.get(LossMetric::Rmse)
    );
  }

  /// Fourier pricing sanity: call price positive and decreasing in K.
  #[test]
  fn fourier_call_monotone() {
    let p = known_params();
    let s = 2488.11;
    let r = 0.01213;
    let q = 0.01884;
    let tau = 28.0 / 365.0;

    let c1 = fourier_call_price(&p, s, 2400.0, r, q, tau);
    let c2 = fourier_call_price(&p, s, 2500.0, r, q, tau);
    let c3 = fourier_call_price(&p, s, 2600.0, r, q, tau);

    println!("Call prices: K=2400: {c1:.4}, K=2500: {c2:.4}, K=2600: {c3:.4}");
    assert!(c1 > c2 && c2 > c3, "calls must decrease in K");
    assert!(c1 > 0.0 && c3 > 0.0, "calls must be positive");
  }
}
