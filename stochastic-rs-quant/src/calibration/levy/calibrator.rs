use std::cell::RefCell;
use std::rc::Rc;

use levenberg_marquardt::LeastSquaresProblem;
use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::Dyn;
use nalgebra::Owned;

use super::EPS;
use super::loss::{
  default_params, fourier_call_price, fourier_option_price, param_bounds, param_count,
  project_params,
};
use super::types::{LevyCalibrationResult, LevyModelType, MarketSlice};
use crate::CalibrationLossScore;
use crate::LossMetric;
use crate::calibration::CalibrationHistory;

/// Lévy model calibrator via Fourier pricing + Levenberg-Marquardt.
///
/// Source:
/// - Levenberg (1944), https://doi.org/10.1090/qam/10666
/// - Marquardt (1963), https://doi.org/10.1137/0111030
#[derive(Clone)]
pub struct LevyCalibrator {
  /// Lévy model to calibrate.
  pub model_type: LevyModelType,
  /// Underlying spot price.
  pub s: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: f64,
  /// Market data slices (one per maturity).
  pub market_data: Vec<MarketSlice>,
  /// If true, record per-iteration calibration history.
  pub record_history: bool,
  /// Internal: current parameter vector (set by LM).
  pub(super) params: Vec<f64>,
  /// Internal: flattened market prices for the residual vector.
  pub(super) flat_prices: Vec<f64>,
  /// Internal: flattened strikes.
  flat_strikes: Vec<f64>,
  /// Internal: flattened maturities.
  flat_t: Vec<f64>,
  /// Internal: flattened is_call flags.
  flat_is_call: Vec<bool>,
  /// Which loss metrics to compute when recording history.
  pub loss_metrics: &'static [LossMetric],
  /// History of iterations.
  calibration_history: Rc<RefCell<Vec<CalibrationHistory<Vec<f64>>>>>,
}

impl LevyCalibrator {
  pub fn new(
    model_type: LevyModelType,
    s: f64,
    r: f64,
    q: f64,
    market_data: Vec<MarketSlice>,
  ) -> Self {
    let (flat_prices, flat_strikes, flat_t, flat_is_call) = Self::flatten(&market_data);
    let params = default_params(model_type);

    Self {
      model_type,
      s,
      r,
      q,
      market_data,
      record_history: false,
      params,
      flat_prices,
      flat_strikes,
      flat_t,
      flat_is_call,
      loss_metrics: &LossMetric::ALL,
      calibration_history: Rc::new(RefCell::new(Vec::new())),
    }
  }

  fn flatten(market_data: &[MarketSlice]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<bool>) {
    let mut prices = Vec::new();
    let mut strikes = Vec::new();
    let mut ts = Vec::new();
    let mut is_call = Vec::new();
    for slice in market_data {
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

  pub(super) fn solve(&self, initial_params: Option<Vec<f64>>) -> LevyCalibrationResult {
    let mut problem = self.clone();
    if let Some(p) = initial_params {
      assert_eq!(
        p.len(),
        param_count(self.model_type),
        "initial_params length must match model parameter count"
      );
      problem.params = p;
    }
    project_params(problem.model_type, &mut problem.params);

    let (result, report) = LevenbergMarquardt::new().minimize(problem);

    let final_params = result.params.clone();
    let c_model = result.compute_model_prices();
    let loss =
      CalibrationLossScore::compute_selected(&result.flat_prices, &c_model, result.loss_metrics);

    LevyCalibrationResult {
      params: final_params,
      model_type: self.model_type,
      loss,
      converged: report.termination.was_successful(),
      iterations: report.number_of_evaluations,
    }
  }

  fn compute_model_prices(&self) -> Vec<f64> {
    (0..self.flat_prices.len())
      .map(|i| {
        fourier_option_price(
          self.model_type,
          &self.params,
          self.s,
          self.flat_strikes[i],
          self.r,
          self.q,
          self.flat_t[i],
          self.flat_is_call[i],
        )
      })
      .collect()
  }

  fn effective_params(&self) -> Vec<f64> {
    let mut p = self.params.clone();
    project_params(self.model_type, &mut p);
    p
  }

  fn numeric_jacobian(&self) -> DMatrix<f64> {
    let n = self.flat_prices.len();
    let m = param_count(self.model_type);
    let bounds = param_bounds(self.model_type);
    let mut j_mat = DMatrix::zeros(n, m);
    let base = self.effective_params();

    for col in 0..m {
      let x = base[col];
      let h = 1e-5_f64.max(1e-3 * x.abs());

      let mut p_plus = base.clone();
      let mut p_minus = base.clone();
      p_plus[col] = (x + h).clamp(bounds[col].0, bounds[col].1);
      p_minus[col] = (x - h).clamp(bounds[col].0, bounds[col].1);
      project_params(self.model_type, &mut p_plus);
      project_params(self.model_type, &mut p_minus);

      let actual_h = p_plus[col] - p_minus[col];
      if actual_h.abs() < EPS {
        continue;
      }

      for i in 0..n {
        let f_plus = fourier_option_price(
          self.model_type,
          &p_plus,
          self.s,
          self.flat_strikes[i],
          self.r,
          self.q,
          self.flat_t[i],
          self.flat_is_call[i],
        );
        let f_minus = fourier_option_price(
          self.model_type,
          &p_minus,
          self.s,
          self.flat_strikes[i],
          self.r,
          self.q,
          self.flat_t[i],
          self.flat_is_call[i],
        );
        // residual = market - model, so d(residual)/dparam = -d(model)/dparam
        j_mat[(i, col)] = -(f_plus - f_minus) / actual_h;
      }
    }

    j_mat
  }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for LevyCalibrator {
  type JacobianStorage = Owned<f64, Dyn, Dyn>;
  type ParameterStorage = Owned<f64, Dyn>;
  type ResidualStorage = Owned<f64, Dyn>;

  fn set_params(&mut self, params: &DVector<f64>) {
    let mut p: Vec<f64> = params.as_slice().to_vec();
    project_params(self.model_type, &mut p);
    self.params = p;
  }

  fn params(&self) -> DVector<f64> {
    DVector::from_vec(self.effective_params())
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    let c_model = self.compute_model_prices();
    let n = self.flat_prices.len();

    if self.record_history {
      self
        .calibration_history
        .borrow_mut()
        .push(CalibrationHistory {
          residuals: DVector::from_iterator(
            n,
            self
              .flat_prices
              .iter()
              .zip(c_model.iter())
              .map(|(m, d)| m - d),
          ),
          call_put: self
            .flat_prices
            .iter()
            .enumerate()
            .map(|(i, _)| {
              let call = fourier_call_price(
                self.model_type,
                &self.params,
                self.s,
                self.flat_strikes[i],
                self.r,
                self.q,
                self.flat_t[i],
              );
              let put = call - self.s * (-self.q * self.flat_t[i]).exp()
                + self.flat_strikes[i] * (-self.r * self.flat_t[i]).exp();
              (call, put.max(0.0))
            })
            .collect::<Vec<(f64, f64)>>()
            .into(),
          params: self.params.clone(),
          loss_scores: CalibrationLossScore::compute_selected(
            &self.flat_prices,
            &c_model,
            self.loss_metrics,
          ),
        });
    }

    let mut residuals = DVector::zeros(n);
    for i in 0..n {
      residuals[i] = self.flat_prices[i] - c_model[i];
    }

    Some(residuals)
  }

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    Some(self.numeric_jacobian())
  }
}
