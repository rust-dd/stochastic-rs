use std::cell::RefCell;

use levenberg_marquardt::LeastSquaresProblem;
use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::Dyn;
use nalgebra::Owned;

use crate::quant::calibration::CalibrationHistory;
use crate::quant::loss;
use crate::quant::pricing::bsm::BSMCoc;
use crate::quant::pricing::bsm::BSMPricer;
use crate::quant::CalibrationLossScore;
use crate::quant::OptionType;
use crate::traits::PricerExt;

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

#[derive(Clone)]
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

impl BSMCalibrator {
  pub fn new(
    params: BSMParams,
    c_market: DVector<f64>,
    s: DVector<f64>,
    k: DVector<f64>,
    r: f64,
    r_d: Option<f64>,
    r_f: Option<f64>,
    q: Option<f64>,
    tau: f64,
    option_type: OptionType,
  ) -> Self {
    Self {
      params,
      c_market,
      s,
      k,
      r,
      r_d,
      r_f,
      q,
      tau,
      option_type,
      calibration_history: RefCell::new(Vec::new()),
      derivates: RefCell::new(Vec::new()),
    }
  }
}

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

  pub fn set_initial_guess(&mut self, params: BSMParams) {
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
    let n = self.c_market.len();
    let mut c_model = DVector::zeros(n);
    let mut vegas: Vec<f64> = Vec::with_capacity(n);
    let mut derivates = Vec::new();

    for (idx, _) in self.c_market.iter().enumerate() {
      let pricer = BSMPricer::new(
        self.s[idx],
        self.params.v,
        self.k[idx],
        self.r,
        self.r_d,
        self.r_f,
        self.q,
        Some(self.tau),
        None,
        None,
        self.option_type,
        BSMCoc::Bsm1973,
      );
      let (call, put) = pricer.calculate_call_put();

      match self.option_type {
        OptionType::Call => c_model[idx] = call,
        OptionType::Put => c_model[idx] = put,
      }

      // Collect vega for vega-weighted residuals (calibration in vol space)
      let vega = pricer.vega().abs().max(1e-8);
      vegas.push(vega);

      self
        .calibration_history
        .borrow_mut()
        .push(CalibrationHistory {
          residuals: c_model.clone() - self.c_market.clone(),
          call_put: vec![(call, put)].into(),
          params: self.params.clone(),
          loss_scores: CalibrationLossScore {
            mae: loss::mae(self.c_market.as_slice(), c_model.as_slice()),
            mse: loss::mse(self.c_market.as_slice(), c_model.as_slice()),
            rmse: loss::rmse(self.c_market.as_slice(), c_model.as_slice()),
            mpe: loss::mpe(self.c_market.as_slice(), c_model.as_slice()),
            mape: loss::mape(self.c_market.as_slice(), c_model.as_slice()),
            mspe: loss::mspe(self.c_market.as_slice(), c_model.as_slice()),
            rmspe: loss::rmspe(self.c_market.as_slice(), c_model.as_slice()),
            mre: loss::mre(self.c_market.as_slice(), c_model.as_slice()),
            mrpe: loss::mrpe(self.c_market.as_slice(), c_model.as_slice()),
          },
        });
      derivates.push(pricer.derivatives());
    }

    let _ = std::mem::replace(&mut *self.derivates.borrow_mut(), derivates);

    // Vega-weighted residuals approximate minimizing implied vol differences
    let mut residuals = DVector::zeros(n);
    for i in 0..n {
      residuals[i] = (c_model[i] - self.c_market[i]) / vegas[i];
    }

    Some(residuals)
  }

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    // For vega-weighted residuals r = (C_model - C_mkt)/Vega,
    // dr/dsigma = 1 - r * (Vomma / Vega)
    let n = self.c_market.len();
    let mut J = DMatrix::zeros(n, 1);

    for idx in 0..n {
      let pricer = BSMPricer::new(
        self.s[idx],
        self.params.v,
        self.k[idx],
        self.r,
        self.r_d,
        self.r_f,
        self.q,
        Some(self.tau),
        None,
        None,
        self.option_type,
        BSMCoc::Bsm1973,
      );

      let (call, put) = pricer.calculate_call_put();
      let c_model_i = match self.option_type {
        OptionType::Call => call,
        OptionType::Put => put,
      };

      let vega = pricer.vega().abs().max(1e-8);
      let vomma = pricer.vomma();
      let r_i = (c_model_i - self.c_market[idx]) / vega;

      J[(idx, 0)] = 1.0 - r_i * (vomma / vega);
    }

    Some(J)
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
