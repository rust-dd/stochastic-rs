use std::cell::RefCell;

use impl_new_derive::ImplNew;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{DMatrix, DVector, Dyn, Owned};

use crate::quant::{
  calibration::CalibrationHistory,
  pricing::bsm::{BSMCoc, BSMPricer},
  r#trait::{CalibrationLossExt, PricerExt},
  CalibrationLossScore, OptionType,
};

#[derive(Clone, Debug)]
pub struct BSMParams {
  /// Implied volatility
  pub v: f64,
}

impl From<BSMParams> for DVector<f64> {
  fn from(params: BSMParams) -> Self {
    DVector::from_vec(vec![params.v])
  }
}

impl From<DVector<f64>> for BSMParams {
  fn from(params: DVector<f64>) -> Self {
    BSMParams { v: params[0] }
  }
}

/// A calibrator.
#[derive(ImplNew, Clone)]
pub struct BSMCalibrator {
  /// Params to calibrate.
  pub params: BSMParams,
  /// Option prices from the market.
  pub c_market: DVector<f64>,
  /// Asset price vector.
  pub s: DVector<f64>,
  /// Strike price vector.
  pub k: DVector<f64>,
  /// Risk-free rate.
  pub r: f64,
  /// Domestic risk-free rate
  pub r_d: Option<f64>,
  /// Foreign risk-free rate
  pub r_f: Option<f64>,
  /// Dividend yield.
  pub q: Option<f64>,
  /// Time to maturity.
  pub tau: f64,
  /// Option type
  pub option_type: OptionType,
  /// Levenberg-Marquardt algorithm residauls.
  calibration_history: RefCell<Vec<CalibrationHistory<BSMParams>>>,
  /// Derivate matrix.
  derivates: RefCell<Vec<Vec<f64>>>,
}

impl CalibrationLossExt for BSMCalibrator {}

impl BSMCalibrator {
  pub fn calibrate(&self) {
    println!("Initial guess: {:?}", self.params);

    let (result, ..) = LevenbergMarquardt::new().minimize(self.clone());

    // Print the c_market
    println!("Market prices: {:?}", self.c_market);

    let residuals = result.residuals().unwrap();

    // Print the c_model
    println!("Model prices: {:?}", self.c_market.clone() + residuals);

    // Print the result of the calibration
    println!("Calibration report: {:?}", result.params);
  }

  pub fn set_intial_guess(&mut self, params: BSMParams) {
    self.params = params;
  }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for BSMCalibrator {
  type JacobianStorage = Owned<f64, Dyn, Dyn>;
  type ParameterStorage = Owned<f64, Dyn>;
  type ResidualStorage = Owned<f64, Dyn>;

  fn set_params(&mut self, params: &DVector<f64>) {
    self.params = BSMParams::from(params.clone());
  }

  fn params(&self) -> DVector<f64> {
    self.params.clone().into()
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    let mut c_model = DVector::zeros(self.c_market.len());
    let mut derivates = Vec::new();

    for (idx, _) in self.c_market.iter().enumerate() {
      let pricer = BSMPricer::new(
        self.s[idx],
        self.params.v,
        self.k[idx],
        self.r,
        None,
        None,
        self.q,
        Some(self.tau),
        None,
        None,
        self.option_type,
        BSMCoc::BSM1973,
      );
      let (call, put) = pricer.calculate_call_put();

      match self.option_type {
        OptionType::Call => c_model[idx] = call,
        OptionType::Put => c_model[idx] = put,
      }

      self
        .calibration_history
        .borrow_mut()
        .push(CalibrationHistory {
          residuals: c_model.clone() - self.c_market.clone(),
          call_put: vec![(call, put)].into(),
          params: self.params.clone().into(),
          loss_scores: CalibrationLossScore {
            mae: self.mae(&self.c_market, &c_model),
            mse: self.mse(&self.c_market, &c_model),
            rmse: self.rmse(&self.c_market, &c_model),
            mpe: self.mpe(&self.c_market, &c_model),
            mape: self.mape(&self.c_market, &c_model),
            mspe: self.mspe(&self.c_market, &c_model),
            rmspe: self.rmspe(&self.c_market, &c_model),
            mre: self.mre(&self.c_market, &c_model),
            mrpe: self.mrpe(&self.c_market, &c_model),
          },
        });
      derivates.push(pricer.derivatives());
    }

    let _ = std::mem::replace(&mut *self.derivates.borrow_mut(), derivates);
    Some(c_model - self.c_market.clone())
  }

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    let derivates = self.derivates.borrow();
    let derivates = derivates.iter().flatten().cloned().collect::<Vec<f64>>();

    // The Jacobian matrix is a matrix of partial derivatives
    // of the residuals with respect to the parameters.
    let jacobian = DMatrix::from_vec(derivates.len() / 5, 5, derivates);

    Some(jacobian)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_calibrate() {
    let s = vec![
      425.73, 425.73, 425.73, 425.67, 425.68, 425.65, 425.65, 425.68, 425.65, 425.16, 424.78,
      425.19,
    ];

    let k = vec![
      395.0, 400.0, 405.0, 410.0, 415.0, 420.0, 425.0, 430.0, 435.0, 440.0, 445.0, 450.0,
    ];

    let c_market = vec![
      30.75, 25.88, 21.00, 16.50, 11.88, 7.69, 4.44, 2.10, 0.78, 0.25, 0.10, 0.10,
    ];

    let r = 0.05;
    let r_d = None;
    let r_f = None;
    let q = None;
    let tau = 1.0;
    let option_type = OptionType::Call;

    let calibrator = BSMCalibrator::new(
      BSMParams { v: 0.2 },
      c_market.into(),
      s.into(),
      k.into(),
      r,
      r_d,
      r_f,
      q,
      tau,
      option_type,
    );

    calibrator.calibrate();
  }
}
