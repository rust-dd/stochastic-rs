//! # SABR
//!
//! Price-based SABR calibrator using Levenberg-Marquardt.
//!
//! **Reference:** P. S. Hagan, D. Kumar, A. S. Lesniewski, D. E. Woodward,
//! *Managing Smile Risk*, Wilmott Magazine, pp. 84–108, 2002.
//!
//! $$
//! dF_t=\alpha_t F_t^\beta dW_t^1,\quad d\alpha_t=\nu\alpha_t dW_t^2,\ d\langle W^1,W^2\rangle_t=\rho dt
//! $$
//!
use std::cell::RefCell;
use std::rc::Rc;

use levenberg_marquardt::LeastSquaresProblem;
use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::Dyn;
use nalgebra::Owned;

use crate::CalibrationLossScore;
use crate::LossMetric;
use crate::OptionType;
use crate::calibration::CalibrationHistory;
use crate::pricing::sabr::SabrPricer;
use crate::traits::PricerExt;

const RHO_BOUND: f64 = 0.9999;
const ALPHA_MIN: f64 = 1e-6;
const NU_MIN: f64 = 1e-6;

/// Calibration result for the SABR model.
#[derive(Clone, Debug)]
pub struct SabrCalibrationResult {
  pub alpha: f64,
  pub beta: f64,
  pub nu: f64,
  pub rho: f64,
  /// Calibration loss metrics.
  pub loss: CalibrationLossScore,
  /// Whether the optimiser converged.
  pub converged: bool,
}

impl crate::traits::ToModel for SabrCalibrationResult {
  fn to_model(&self, _r: f64, _q: f64) -> Box<dyn crate::traits::ModelPricer> {
    Box::new(SabrCalibrationResult::to_model(self))
  }
}

impl SabrCalibrationResult {
  /// Convert to a [`SabrModel`] for pricing / vol surface generation.
  pub fn to_model(&self) -> crate::pricing::sabr::SabrModel {
    crate::pricing::sabr::SabrModel {
      alpha: self.alpha,
      beta: self.beta,
      nu: self.nu,
      rho: self.rho,
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub struct SabrParams {
  /// Model shape/loading parameter.
  pub alpha: f64,
  /// CEV exponent (0 = normal, 1 = lognormal).
  pub beta: f64,
  /// Volatility-of-volatility parameter.
  pub nu: f64,
  /// Correlation parameter.
  pub rho: f64,
}

impl SabrParams {
  pub fn project_in_place(&mut self) {
    self.alpha = self.alpha.abs().max(ALPHA_MIN);
    self.beta = self.beta.clamp(0.0, 1.0);
    self.nu = self.nu.abs().max(NU_MIN);
    self.rho = self.rho.clamp(-RHO_BOUND, RHO_BOUND);
  }
  pub fn projected(mut self) -> Self {
    self.project_in_place();
    self
  }
}

impl From<SabrParams> for DVector<f64> {
  fn from(p: SabrParams) -> Self {
    DVector::from_vec(vec![p.alpha, p.nu, p.rho])
  }
}

impl From<DVector<f64>> for SabrParams {
  fn from(v: DVector<f64>) -> Self {
    SabrParams {
      alpha: v[0],
      beta: 1.0,
      nu: v[1],
      rho: v[2],
    }
  }
}

#[derive(Clone)]
pub struct SabrCalibrator {
  /// Model parameter set (input or calibrated output).
  pub params: Option<SabrParams>,
  /// Observed market option prices used for calibration.
  pub c_market: DVector<f64>,
  /// Underlying spot/forward level.
  pub s: DVector<f64>,
  /// Strike level.
  pub k: DVector<f64>,
  /// Risk-free rate used for discounting.
  pub r: f64,
  /// Dividend yield / convenience yield.
  pub q: Option<f64>,
  /// Time-to-maturity in years.
  pub tau: f64,
  /// Option direction (call/put).
  pub option_type: OptionType,
  /// If true, stores optimization parameter history.
  pub record_history: bool,
  /// Which loss metrics to compute when recording history.
  pub loss_metrics: &'static [LossMetric],
  calibration_history: Rc<RefCell<Vec<CalibrationHistory<SabrParams>>>>,
}

impl SabrCalibrator {
  pub fn new(
    params: Option<SabrParams>,
    c_market: DVector<f64>,
    s: DVector<f64>,
    k: DVector<f64>,
    r: f64,
    q: Option<f64>,
    tau: f64,
    option_type: OptionType,
    record_history: bool,
  ) -> Self {
    Self {
      params,
      c_market,
      s,
      k,
      r,
      q,
      tau,
      option_type,
      record_history,
      loss_metrics: &LossMetric::ALL,
      calibration_history: Rc::new(RefCell::new(Vec::new())),
    }
  }
}

impl SabrCalibrator {
  pub fn calibrate(&self) -> SabrCalibrationResult {
    let mut problem = self.clone();
    problem.ensure_initial_guess();

    let (result, report) = LevenbergMarquardt::new().minimize(problem);
    let converged = report.termination.was_successful();
    let p = result.effective_params();
    let c_model = result.compute_model_prices_for(&p);
    let loss = CalibrationLossScore::compute_selected(
      result.c_market.as_slice(),
      c_model.as_slice(),
      result.loss_metrics,
    );

    SabrCalibrationResult {
      alpha: p.alpha,
      beta: p.beta,
      nu: p.nu,
      rho: p.rho,
      loss,
      converged,
    }
  }

  pub fn set_initial_guess(&mut self, params: SabrParams) {
    self.params = Some(params.projected());
  }
  pub fn set_record_history(&mut self, record: bool) {
    self.record_history = record;
  }
  pub fn history(&self) -> Vec<CalibrationHistory<SabrParams>> {
    self.calibration_history.borrow().clone()
  }

  fn ensure_initial_guess(&mut self) {
    if self.params.is_none() {
      self.params = Some(
        SabrParams {
          alpha: 0.2,
          beta: 1.0,
          nu: 0.8,
          rho: 0.0,
        }
        .projected(),
      );
    }
  }

  fn effective_params(&self) -> SabrParams {
    if let Some(p) = &self.params {
      return (*p).projected();
    }
    SabrParams {
      alpha: 0.2,
      beta: 1.0,
      nu: 0.8,
      rho: 0.0,
    }
    .projected()
  }

  fn compute_model_prices_for(&self, p: &SabrParams) -> DVector<f64> {
    let mut c_model = DVector::zeros(self.c_market.len());
    for i in 0..self.c_market.len() {
      let pr = SabrPricer::new(
        self.s[i],
        self.k[i],
        self.r,
        self.q,
        p.alpha,
        p.beta,
        p.nu,
        p.rho,
        Some(self.tau),
        None,
        None,
      );
      let (call, put) = pr.calculate_call_put();
      c_model[i] = match self.option_type {
        OptionType::Call => call,
        OptionType::Put => put,
      };
    }
    c_model
  }

  fn residuals_for(&self, p: &SabrParams) -> DVector<f64> {
    self.c_market.clone() - self.compute_model_prices_for(p)
  }

  fn numeric_jacobian(&self, p: &SabrParams) -> DMatrix<f64> {
    let n = self.c_market.len();
    let m = 3usize; // alpha, nu, rho
    let base: DVector<f64> = (*p).into();
    let mut J = DMatrix::zeros(n, m);
    for col in 0..m {
      let x = base[col];
      let mut h = 1e-5_f64.max(1e-3 * x.abs());
      let mut p_plus = *p;
      let mut p_minus = *p;
      match col {
        0 => {
          p_plus.alpha = (x + h).abs().max(ALPHA_MIN);
          p_minus.alpha = (x - h).abs().max(ALPHA_MIN);
        }
        1 => {
          p_plus.nu = (x + h).abs().max(NU_MIN);
          p_minus.nu = (x - h).abs().max(NU_MIN);
        }
        2 => {
          let clamp = |y: f64| y.clamp(-RHO_BOUND, RHO_BOUND);
          p_plus.rho = clamp(x + h);
          p_minus.rho = clamp(x - h);
          if (p_plus.rho - p_minus.rho).abs() < 0.5 * h {
            h = 1e-4;
            p_plus.rho = clamp(x + h);
            p_minus.rho = clamp(x - h);
          }
        }
        _ => unreachable!(),
      }
      p_plus.project_in_place();
      p_minus.project_in_place();
      let r_plus = self.residuals_for(&p_plus);
      let r_minus = self.residuals_for(&p_minus);
      let diff = (r_plus - r_minus) / (2.0 * h);
      for row in 0..n {
        J[(row, col)] = diff[row];
      }
    }
    J
  }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for SabrCalibrator {
  type JacobianStorage = Owned<f64, Dyn, Dyn>;
  type ParameterStorage = Owned<f64, Dyn>;
  type ResidualStorage = Owned<f64, Dyn>;

  fn set_params(&mut self, params: &DVector<f64>) {
    let beta = self.effective_params().beta;
    let mut p = SabrParams {
      alpha: params[0],
      beta,
      nu: params[1],
      rho: params[2],
    };
    p.project_in_place();
    self.params = Some(p);
  }
  fn params(&self) -> DVector<f64> {
    self.effective_params().into()
  }
  fn residuals(&self) -> Option<DVector<f64>> {
    let p = self.effective_params();
    let c_model = self.compute_model_prices_for(&p);
    if self.record_history {
      self
        .calibration_history
        .borrow_mut()
        .push(CalibrationHistory {
          residuals: self.c_market.clone() - c_model.clone(),
          call_put: self
            .c_market
            .iter()
            .enumerate()
            .map(|(i, _)| {
              let pr = SabrPricer::new(
                self.s[i],
                self.k[i],
                self.r,
                self.q,
                p.alpha,
                p.beta,
                p.nu,
                p.rho,
                Some(self.tau),
                None,
                None,
              );
              pr.calculate_call_put()
            })
            .collect::<Vec<(f64, f64)>>()
            .into(),
          params: p,
          loss_scores: CalibrationLossScore::compute_selected(
            self.c_market.as_slice(),
            c_model.as_slice(),
            self.loss_metrics,
          ),
        });
    }
    Some(self.c_market.clone() - c_model)
  }
  fn jacobian(&self) -> Option<DMatrix<f64>> {
    Some(self.numeric_jacobian(&self.effective_params()))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_sabr_calibrate_price_based() {
    let s = vec![100.0; 8];
    let k = vec![80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0];
    let r = 0.02;
    let q = Some(0.01);
    let tau = 0.5;

    let true_p = SabrParams {
      alpha: 0.2,
      beta: 1.0,
      nu: 0.6,
      rho: -0.4,
    };

    // Build synthetic market prices
    let mut c_market = Vec::new();
    for &kk in &k {
      let pr = SabrPricer::new(
        100.0,
        kk,
        r,
        q,
        true_p.alpha,
        true_p.beta,
        true_p.nu,
        true_p.rho,
        Some(tau),
        None,
        None,
      );
      let (call, _) = pr.calculate_call_put();
      c_market.push(call);
    }

    let calibrator = SabrCalibrator::new(
      Some(SabrParams {
        alpha: 0.15,
        beta: 1.0,
        nu: 0.8,
        rho: 0.0,
      }),
      c_market.clone().into(),
      s.clone().into(),
      k.clone().into(),
      r,
      q,
      tau,
      OptionType::Call,
      true,
    );

    calibrator.calibrate();
  }
}
