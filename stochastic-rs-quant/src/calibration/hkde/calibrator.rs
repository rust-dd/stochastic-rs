use std::cell::RefCell;
use std::rc::Rc;

use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DVector;

use super::loss::compute_sqrt_weights;
use super::params::HKDEParams;
use super::result::HKDECalibrationResult;
use crate::CalibrationLossScore;
use crate::LossMetric;
use crate::OptionType;
use crate::calibration::CalibrationHistory;

/// Hkde least-squares calibrator using Levenberg-Marquardt.
///
/// Source:
/// - Levenberg (1944), https://doi.org/10.1090/qam/10666
/// - Marquardt (1963), https://doi.org/10.1137/0111030
/// - Agazzotti et al. (2025), arXiv: 2502.13824
#[derive(Clone)]
pub struct HKDECalibrator {
  /// Current parameter iterate. `None` triggers the fallback initial guess.
  pub params: Option<HKDEParams>,
  /// Market option prices (flattened across all maturities).
  pub c_market: DVector<f64>,
  /// Underlying spot per quote.
  pub s: DVector<f64>,
  /// Strike per quote.
  pub k: DVector<f64>,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: Option<f64>,
  /// Time to maturity per quote (flattened). Supports joint multi-maturity
  /// calibration.
  pub flat_t: Vec<f64>,
  /// Option type.
  pub option_type: OptionType,
  /// Precomputed vega weights $\sqrt{w_j^{(n)}}$ applied to each residual
  /// (Eq. 13 of the paper). Length equals the number of quotes.
  pub sqrt_weights: Vec<f64>,
  /// If true, record per-iteration calibration history.
  pub record_history: bool,
  /// Which loss metrics to compute when recording history.
  pub loss_metrics: &'static [LossMetric],
  pub(super) calibration_history: Rc<RefCell<Vec<CalibrationHistory<HKDEParams>>>>,
}

impl HKDECalibrator {
  /// Create a calibrator for a single maturity slice.
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    params: Option<HKDEParams>,
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

    let flat_t = vec![tau; n];
    let q_val = q.unwrap_or(0.0);
    let sqrt_weights = compute_sqrt_weights(
      s.as_slice(),
      k.as_slice(),
      &flat_t,
      r,
      q_val,
      c_market.as_slice(),
      option_type,
    );

    Self {
      params,
      c_market,
      s,
      k,
      r,
      q,
      flat_t,
      option_type,
      sqrt_weights,
      record_history,
      loss_metrics: &LossMetric::ALL,
      calibration_history: Rc::new(RefCell::new(Vec::new())),
    }
  }

  /// Create a calibrator from multiple maturity slices for joint surface
  /// calibration. Mirrors the API of the Heston / SVJ / BSM calibrators.
  pub fn from_slices(
    params: Option<HKDEParams>,
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

    let q_val = q.unwrap_or(0.0);
    let sqrt_weights = compute_sqrt_weights(
      &flat_s,
      &flat_strikes,
      &flat_t,
      r,
      q_val,
      &flat_prices,
      option_type,
    );

    Self {
      params,
      c_market: DVector::from_vec(flat_prices),
      s: DVector::from_vec(flat_s),
      k: DVector::from_vec(flat_strikes),
      r,
      q,
      flat_t,
      option_type,
      sqrt_weights,
      record_history,
      loss_metrics: &LossMetric::ALL,
      calibration_history: Rc::new(RefCell::new(Vec::new())),
    }
  }

  /// Run the calibration. If `initial_params` is `Some`, it overrides the
  /// calibrator's current initial guess.
  pub fn calibrate(&self, initial_params: Option<HKDEParams>) -> HKDECalibrationResult {
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

    HKDECalibrationResult {
      v0: p.v0,
      kappa: p.kappa,
      theta: p.theta,
      sigma_v: p.sigma_v,
      rho: p.rho,
      lambda: p.lambda,
      p_up: p.p_up,
      eta1: p.eta1,
      eta2: p.eta2,
      loss,
      converged: report.termination.was_successful(),
    }
  }

  pub fn set_initial_guess(&mut self, params: HKDEParams) {
    self.params = Some(params.projected());
  }

  pub fn set_record_history(&mut self, record: bool) {
    self.record_history = record;
  }

  pub fn history(&self) -> Vec<CalibrationHistory<HKDEParams>> {
    self.calibration_history.borrow().clone()
  }

  pub(super) fn fallback_params() -> HKDEParams {
    HKDEParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma_v: 0.3,
      rho: -0.5,
      lambda: 0.5,
      p_up: 0.4,
      eta1: 10.0,
      eta2: 5.0,
    }
    .projected()
  }

  pub(super) fn ensure_initial_guess(&mut self) {
    if self.params.is_none() {
      self.params = Some(Self::fallback_params());
    }
  }

  pub(super) fn effective_params(&self) -> HKDEParams {
    if let Some(p) = &self.params {
      return (*p).projected();
    }
    Self::fallback_params()
  }
}

impl crate::traits::Calibrator for HKDECalibrator {
  type InitialGuess = HKDEParams;
  type Params = HKDEParams;
  type Output = HKDECalibrationResult;
  type Error = anyhow::Error;

  fn calibrate(&self, initial: Option<Self::InitialGuess>) -> Result<Self::Output, Self::Error> {
    Ok(HKDECalibrator::calibrate(self, initial))
  }
}
