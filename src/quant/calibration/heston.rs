use std::cell::RefCell;

use anyhow::Result;
use impl_new_derive::ImplNew;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{DMatrix, DVector, Dyn, Owned};
use ndarray::Array1;

use crate::{
  quant::{
    pricing::heston::HestonPricer,
    r#trait::{CalibrationLossExt, PricerExt},
    CalibrationLossScore, OptionType,
  },
  stats::mle::nmle_heston,
};

/// Heston model parameters
#[derive(Clone, Debug)]
pub struct HestonParams {
  pub v0: f64,
  pub theta: f64,
  pub rho: f64,
  pub kappa: f64,
  pub sigma: f64,
}

impl From<HestonParams> for DVector<f64> {
  fn from(params: HestonParams) -> Self {
    DVector::from_vec(vec![
      params.v0,
      params.theta,
      params.rho,
      params.kappa,
      params.sigma,
    ])
  }
}

impl From<DVector<f64>> for HestonParams {
  fn from(params: DVector<f64>) -> Self {
    HestonParams {
      v0: params[0],
      theta: params[1],
      rho: params[2],
      kappa: params[3],
      sigma: params[4],
    }
  }
}

#[derive(Clone, Debug)]
pub struct CalibrationHistory<T> {
  pub residuals: DVector<f64>,
  pub call_put: DVector<(f64, f64)>,
  pub params: T,
  pub loss_scores: CalibrationLossScore,
}

/// A calibrator.
#[derive(ImplNew, Clone)]
pub struct HestonCalibrator {
  /// Params to calibrate.
  pub params: HestonParams,
  /// Option prices from the market.
  pub c_market: DVector<f64>,
  /// Asset price vector.
  pub s: DVector<f64>,
  /// Strike price vector.
  pub k: DVector<f64>,
  /// Time to maturity.
  pub tau: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: Option<f64>,
  /// Option type
  pub option_type: OptionType,
  /// Levenberg-Marquardt algorithm residauls.
  calibration_history: RefCell<Vec<CalibrationHistory<HestonParams>>>,
  /// Derivate matrix.
  derivates: RefCell<Vec<Vec<f64>>>,
}

impl CalibrationLossExt for HestonCalibrator {}

impl HestonCalibrator {
  pub fn calibrate(&self) -> Result<Vec<CalibrationHistory<HestonParams>>> {
    println!("Initial guess: {:?}", self.params);

    let (result, ..) = LevenbergMarquardt::new().minimize(self.clone());

    // Print the c_market
    println!("Market prices: {:?}", self.c_market);

    let residuals = result.residuals().unwrap();

    // Print the c_model
    println!("Model prices: {:?}", self.c_market.clone() + residuals);

    // Print the result of the calibration
    println!("Calibration report: {:?}", result.params);

    let calibration_history = result.calibration_history.borrow().clone();

    Ok(calibration_history)
  }

  /// Initial guess for the calibration
  /// http://scis.scichina.com/en/2018/042202.pdf
  ///
  /// Using NMLE (Normal Maximum Likelihood Estimation) method
  pub fn set_initial_params(&mut self, s: Array1<f64>, v: Array1<f64>, r: f64) {
    self.params = nmle_heston(s, v, r);
  }
}

impl<'a> LeastSquaresProblem<f64, Dyn, Dyn> for HestonCalibrator {
  type JacobianStorage = Owned<f64, Dyn, Dyn>;
  type ParameterStorage = Owned<f64, Dyn>;
  type ResidualStorage = Owned<f64, Dyn>;

  fn set_params(&mut self, params: &DVector<f64>) {
    self.params = HestonParams::from(params.clone());
  }

  fn params(&self) -> DVector<f64> {
    self.params.clone().into()
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    let mut c_model = DVector::zeros(self.c_market.len());
    let mut derivates = Vec::new();

    for (idx, _) in self.c_market.iter().enumerate() {
      let pricer = HestonPricer::new(
        self.s[idx],
        self.params.v0,
        self.k[idx],
        self.r,
        self.q,
        self.params.rho,
        self.params.kappa,
        self.params.theta,
        self.params.sigma,
        None,
        Some(self.tau),
        None,
        None,
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
            mae: self.mae(&c_model, &self.c_market),
            mse: self.mse(&c_model, &self.c_market),
            rmse: self.rmse(&c_model, &self.c_market),
            mpe: self.mpe(&c_model, &self.c_market),
            mape: self.mape(&c_model, &self.c_market),
            mspe: self.mspe(&c_model, &self.c_market),
            rmspe: self.rmspe(&c_model, &self.c_market),
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

  use anyhow::Result;

  #[test]
  fn test_heston_calibrate() -> Result<()> {
    let tau = 24.0 / 365.0;
    println!("Time to maturity: {}", tau);

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

    let v0 = Array1::linspace(0.0, 0.01, 1);

    for v in v0.iter() {
      let calibrator = HestonCalibrator::new(
        HestonParams {
          v0: *v,
          theta: 6.47e-5,
          rho: -1.98e-3,
          kappa: 6.57e-3,
          sigma: 5.09e-4,
        },
        c_market.clone().into(),
        s.clone().into(),
        k.clone().into(),
        tau,
        6.40e-4,
        None,
        OptionType::Call,
      );

      let data = calibrator.calibrate()?;
      println!("Calibration data: {:?}", data);
    }

    Ok(())
  }
}
