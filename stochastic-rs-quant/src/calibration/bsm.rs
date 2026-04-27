//! # Bsm
//!
//! $$
//! C=S_0e^{(b-r)T}N(d_1)-Ke^{-rT}N(d_2),\quad d_{1,2}=\frac{\ln(S_0/K)+(b\pm\tfrac12\sigma^2)T}{\sigma\sqrt T}
//! $$
//!
use std::cell::RefCell;

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
use crate::pricing::bsm::BSMCoc;
use crate::pricing::bsm::BSMPricer;
use crate::traits::PricerExt;

/// Calibration result for the BSM model.
#[derive(Clone, Debug)]
pub struct BSMCalibrationResult {
  /// Calibrated implied volatility.
  pub v: f64,
  /// Calibration loss metrics.
  pub loss: CalibrationLossScore,
  /// Whether the optimiser converged.
  pub converged: bool,
}

impl BSMCalibrationResult {
  /// Convert to a [`BSMFourier`] model for pricing / vol surface generation.
  pub fn to_model(&self, r: f64, q: f64) -> crate::pricing::fourier::BSMFourier {
    crate::pricing::fourier::BSMFourier {
      sigma: self.v,
      r,
      q,
    }
  }
}

impl crate::traits::ToModel for BSMCalibrationResult {
  type Model = crate::pricing::fourier::BSMFourier;
  fn to_model(&self, r: f64, q: f64) -> Self::Model {
    BSMCalibrationResult::to_model(self, r, q)
  }
}

impl crate::traits::CalibrationResult for BSMCalibrationResult {
  fn rmse(&self) -> f64 {
    self.loss.get(LossMetric::Rmse)
  }
  fn converged(&self) -> bool {
    self.converged
  }
  fn loss_score(&self) -> Option<&CalibrationLossScore> {
    Some(&self.loss)
  }
}

impl crate::traits::Calibrator for BSMCalibrator {
  type InitialGuess = BSMParams;
  type Output = BSMCalibrationResult;
  fn calibrate(&self, initial: Option<Self::InitialGuess>) -> Self::Output {
    let mut this = self.clone();
    if let Some(p) = initial {
      this.set_initial_guess(p);
    }
    BSMCalibrator::calibrate(&this)
  }
}

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
  /// Option prices from the market (flattened across all maturities).
  pub c_market: DVector<f64>,
  /// Underlying spot per quote.
  pub s: DVector<f64>,
  /// Strike per quote (flattened).
  pub k: DVector<f64>,
  /// Risk-free rate.
  pub r: f64,
  /// Domestic risk-free rate
  pub r_d: Option<f64>,
  /// Foreign risk-free rate
  pub r_f: Option<f64>,
  /// Dividend yield.
  pub q: Option<f64>,
  /// Time to maturity (kept for the legacy single-tau constructor).
  pub tau: f64,
  /// Time to maturity per quote (flattened). Supports multi-maturity
  /// joint calibration. Always populated — for the single-tau
  /// `BSMCalibrator::new` constructor every entry equals `tau`.
  pub flat_t: Vec<f64>,
  /// Option type
  pub option_type: OptionType,
  /// Which loss metrics to compute when recording history.
  pub loss_metrics: &'static [LossMetric],
  /// Levenberg-Marquardt algorithm residauls.
  calibration_history: RefCell<Vec<CalibrationHistory<BSMParams>>>,
  /// Derivate matrix.
  derivates: RefCell<Vec<Vec<f64>>>,
}

impl BSMCalibrator {
  /// Create a calibrator for a single maturity slice (backwards compatible).
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
    let n = c_market.len();
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
      flat_t: vec![tau; n],
      option_type,
      loss_metrics: &LossMetric::ALL,
      calibration_history: RefCell::new(Vec::new()),
      derivates: RefCell::new(Vec::new()),
    }
  }

  /// Create a calibrator from multiple maturity slices for joint
  /// surface calibration. Mirrors the API of the Heston / SVJ
  /// calibrators so a single chain of `MarketSlice`s can be used to
  /// fit BSM, Heston and Bates side by side.
  pub fn from_slices(
    params: BSMParams,
    slices: &[super::levy::MarketSlice],
    s: f64,
    r: f64,
    r_d: Option<f64>,
    r_f: Option<f64>,
    q: Option<f64>,
    option_type: OptionType,
  ) -> Self {
    let mut flat_prices = Vec::new();
    let mut flat_strikes = Vec::new();
    let mut flat_t = Vec::new();
    let mut flat_s = Vec::new();

    for slice in slices {
      for i in 0..slice.strikes.len() {
        flat_prices.push(slice.prices[i]);
        flat_strikes.push(slice.strikes[i]);
        flat_t.push(slice.t);
        flat_s.push(s);
      }
    }

    Self {
      params,
      c_market: DVector::from_vec(flat_prices),
      s: DVector::from_vec(flat_s),
      k: DVector::from_vec(flat_strikes),
      r,
      r_d,
      r_f,
      q,
      tau: 0.0,
      flat_t,
      option_type,
      loss_metrics: &LossMetric::ALL,
      calibration_history: RefCell::new(Vec::new()),
      derivates: RefCell::new(Vec::new()),
    }
  }
}

impl BSMCalibrator {
  pub fn calibrate(&self) -> BSMCalibrationResult {
    let (result, report) = LevenbergMarquardt::new().minimize(self.clone());
    let converged = report.termination.was_successful();
    let c_model: Vec<f64> = result
      .c_market
      .iter()
      .enumerate()
      .map(|(idx, _)| {
        let pricer = BSMPricer::new(
          result.s[idx],
          result.params.v,
          result.k[idx],
          result.r,
          result.r_d,
          result.r_f,
          result.q,
          Some(result.flat_t[idx]),
          None,
          None,
          result.option_type,
          BSMCoc::Bsm1973,
        );
        let (call, put) = pricer.calculate_call_put();
        match result.option_type {
          OptionType::Call => call,
          OptionType::Put => put,
        }
      })
      .collect();
    let loss = CalibrationLossScore::compute_selected(
      result.c_market.as_slice(),
      &c_model,
      result.loss_metrics,
    );

    BSMCalibrationResult {
      v: result.params.v,
      loss,
      converged,
    }
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
        Some(self.flat_t[idx]),
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
          loss_scores: CalibrationLossScore::compute_selected(
            self.c_market.as_slice(),
            c_model.as_slice(),
            self.loss_metrics,
          ),
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
        Some(self.flat_t[idx]),
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

  #[test]
  fn test_calibrate_from_slices_recovers_constant_sigma() {
    // Generate three synthetic maturity slices from a known constant sigma,
    // then check the joint calibrator recovers it on the whole flattened set.
    use crate::calibration::levy::MarketSlice;

    let s = 100.0_f64;
    let r = 0.03_f64;
    let true_sigma = 0.25_f64;
    let strikes = vec![85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0];

    let make_slice = |t: f64| -> MarketSlice {
      let prices: Vec<f64> = strikes
        .iter()
        .map(|&k| {
          let pricer = BSMPricer::builder(s, true_sigma, k, r)
            .tau(t)
            .coc(BSMCoc::Bsm1973)
            .build();
          let (call, _) = pricer.calculate_call_put();
          call
        })
        .collect();
      MarketSlice {
        strikes: strikes.clone(),
        prices,
        is_call: vec![true; strikes.len()],
        t,
      }
    };

    let slices = vec![make_slice(0.10), make_slice(0.30), make_slice(0.75)];

    let calibrator = BSMCalibrator::from_slices(
      BSMParams { v: 0.4 }, // intentionally far from the truth
      &slices,
      s,
      r,
      None,
      None,
      None,
      OptionType::Call,
    );
    let result = calibrator.calibrate();
    println!(
      "recovered sigma = {:.6}  (truth {:.4})  converged = {}",
      result.v, true_sigma, result.converged
    );
    assert!(
      (result.v - true_sigma).abs() < 1e-3,
      "expected ~{}, got {}",
      true_sigma,
      result.v
    );
  }
}
