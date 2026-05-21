use std::cell::RefCell;
use std::rc::Rc;

use levenberg_marquardt::LeastSquaresProblem;
use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::Dyn;
use nalgebra::Owned;

use super::loss::double_heston_call_price;
use super::params::DoubleHestonParams;
use super::result::DoubleHestonCalibrationResult;
use crate::CalibrationLossScore;
use crate::LossMetric;
use crate::OptionType;
use crate::calibration::CalibrationHistory;

/// Double Heston least-squares calibrator using Levenberg-Marquardt.
#[derive(Clone)]
pub struct DoubleHestonCalibrator {
  pub params: Option<DoubleHestonParams>,
  pub c_market: DVector<f64>,
  pub s: DVector<f64>,
  pub k: DVector<f64>,
  pub r: f64,
  pub q: Option<f64>,
  pub flat_t: Vec<f64>,
  pub option_type: OptionType,
  pub record_history: bool,
  pub loss_metrics: &'static [LossMetric],
  pub(super) calibration_history: Rc<RefCell<Vec<CalibrationHistory<DoubleHestonParams>>>>,
}

impl DoubleHestonCalibrator {
  /// Create a calibrator for a single maturity slice.
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    params: Option<DoubleHestonParams>,
    c_market: DVector<f64>,
    s: DVector<f64>,
    k: DVector<f64>,
    r: f64,
    q: Option<f64>,
    tau: f64,
    option_type: OptionType,
    record_history: bool,
  ) -> Self {
    let n = c_market.len();
    assert_eq!(n, s.len(), "c_market and s must have the same length");
    assert_eq!(n, k.len(), "c_market and k must have the same length");
    assert!(
      tau.is_finite() && tau > 0.0,
      "tau must be a finite positive value"
    );

    Self {
      params,
      c_market,
      s,
      k,
      r,
      q,
      flat_t: vec![tau; n],
      option_type,
      record_history,
      loss_metrics: &LossMetric::ALL,
      calibration_history: Rc::new(RefCell::new(Vec::new())),
    }
  }

  /// Create a calibrator from multiple maturity slices for joint surface calibration.
  pub fn from_slices(
    params: Option<DoubleHestonParams>,
    slices: &[super::super::levy::MarketSlice],
    s: f64,
    r: f64,
    q: Option<f64>,
    option_type: OptionType,
    record_history: bool,
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
      q,
      flat_t,
      option_type,
      record_history,
      loss_metrics: &LossMetric::ALL,
      calibration_history: Rc::new(RefCell::new(Vec::new())),
    }
  }

  pub(super) fn solve(
    &self,
    initial_params: Option<DoubleHestonParams>,
  ) -> DoubleHestonCalibrationResult {
    let mut problem = self.clone();
    if let Some(p) = initial_params {
      problem.params = Some(p.projected());
    }
    problem.ensure_initial_guess();

    let (result, report) = LevenbergMarquardt::new().minimize(problem);

    let p = result.effective_params();
    let c_model = result.compute_model_prices_for(&p);
    let loss = CalibrationLossScore::compute_selected(
      result.c_market.as_slice(),
      c_model.as_slice(),
      result.loss_metrics,
    );

    DoubleHestonCalibrationResult {
      v1_0: p.v1_0,
      kappa1: p.kappa1,
      theta1: p.theta1,
      sigma1: p.sigma1,
      rho1: p.rho1,
      v2_0: p.v2_0,
      kappa2: p.kappa2,
      theta2: p.theta2,
      sigma2: p.sigma2,
      rho2: p.rho2,
      loss,
      converged: report.termination.was_successful(),
    }
  }

  pub fn set_initial_guess(&mut self, params: DoubleHestonParams) {
    self.params = Some(params.projected());
  }

  pub fn set_record_history(&mut self, record: bool) {
    self.record_history = record;
  }

  pub fn history(&self) -> Vec<CalibrationHistory<DoubleHestonParams>> {
    self.calibration_history.borrow().clone()
  }

  /// Fallback: a fast-and-slow factor split centred around realistic
  /// equity-index values (fast factor mean-reverts in ≈ 4 months, slow factor
  /// in ≈ 2 years).
  pub(super) fn fallback_params() -> DoubleHestonParams {
    DoubleHestonParams {
      v1_0: 0.02,
      kappa1: 3.0,
      theta1: 0.02,
      sigma1: 0.3,
      rho1: -0.6,
      v2_0: 0.02,
      kappa2: 0.5,
      theta2: 0.02,
      sigma2: 0.15,
      rho2: -0.3,
    }
    .projected()
  }

  pub(super) fn ensure_initial_guess(&mut self) {
    if self.params.is_none() {
      self.params = Some(Self::fallback_params());
    }
  }

  pub(super) fn effective_params(&self) -> DoubleHestonParams {
    if let Some(p) = &self.params {
      return (*p).projected();
    }
    Self::fallback_params()
  }
}

impl crate::traits::Calibrator for DoubleHestonCalibrator {
  type InitialGuess = DoubleHestonParams;
  type Params = DoubleHestonParams;
  type Output = DoubleHestonCalibrationResult;
  type Error = anyhow::Error;

  fn calibrate(&self, initial: Option<Self::InitialGuess>) -> Result<Self::Output, Self::Error> {
    Ok(self.solve(initial))
  }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for DoubleHestonCalibrator {
  type JacobianStorage = Owned<f64, Dyn, Dyn>;
  type ParameterStorage = Owned<f64, Dyn>;
  type ResidualStorage = Owned<f64, Dyn>;

  fn set_params(&mut self, params: &DVector<f64>) {
    let p = DoubleHestonParams::from(params.clone()).projected();
    self.params = Some(p);
  }

  fn params(&self) -> DVector<f64> {
    self.effective_params().into()
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    let params_eff = self.effective_params();
    let c_model = self.compute_model_prices_for(&params_eff);

    if self.record_history {
      let q_val = self.q.unwrap_or(0.0);
      self
        .calibration_history
        .borrow_mut()
        .push(CalibrationHistory {
          residuals: self.c_market.clone() - c_model.clone(),
          call_put: self
            .c_market
            .iter()
            .enumerate()
            .map(|(idx, _)| {
              let tau = self.flat_t[idx];
              let call =
                double_heston_call_price(&params_eff, self.s[idx], self.k[idx], self.r, q_val, tau);
              let put =
                call - self.s[idx] * (-q_val * tau).exp() + self.k[idx] * (-self.r * tau).exp();
              (call.max(0.0), put.max(0.0))
            })
            .collect::<Vec<(f64, f64)>>()
            .into(),
          params: params_eff,
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
