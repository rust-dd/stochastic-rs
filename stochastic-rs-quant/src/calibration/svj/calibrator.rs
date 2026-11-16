use std::cell::RefCell;
use std::rc::Rc;

use ndarray::Array1;

use super::SvjCalibrationResult;
use super::SvjParams;
use crate::CalibrationLossScore;
use crate::LossMetric;
use crate::OptionType;
use crate::QuantError;
use crate::calibration::CalibrationHistory;
use crate::calibration::least_squares::LmOptions;
use crate::calibration::least_squares::minimize;

pub(super) const KAPPA_MIN: f64 = 1e-3;

pub(super) const THETA_MIN: f64 = 1e-8;

pub(super) const SIGMA_V_MIN: f64 = 1e-8;

/// SVJ (Bates) least-squares calibrator using Levenberg-Marquardt.
///
/// Source:
/// - Levenberg (1944), <https://doi.org/10.1090/qam/10666>
/// - Marquardt (1963), <https://doi.org/10.1137/0111030>
/// - Bates (1996), <https://doi.org/10.1093/rfs/9.1.69>
#[derive(Clone)]
pub struct SvjCalibrator {
  /// Params to calibrate.
  pub params: Option<SvjParams>,
  /// Option prices from the market (flattened across all maturities).
  pub c_market: Array1<f64>,
  /// Underlying spot per quote.
  pub s: Array1<f64>,
  /// Strikes per quote (flattened).
  pub k: Array1<f64>,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: Option<f64>,
  /// Time to maturity per quote (flattened). Supports multi-maturity joint calibration.
  pub flat_t: Vec<f64>,
  /// Whether the market quotes are call or put prices.
  pub option_type: OptionType,
  /// If true, record per-iteration calibration history.
  pub record_history: bool,
  /// Which loss metrics to compute when recording history.
  pub loss_metrics: &'static [LossMetric],
  pub(super) calibration_history: Rc<RefCell<Vec<CalibrationHistory<SvjParams>>>>,
}

impl crate::traits::Calibrator for SvjCalibrator {
  type Params = SvjParams;
  type Output = SvjCalibrationResult;
  type Error = QuantError;

  fn calibrate(&self, initial: Option<Self::Params>) -> Result<Self::Output, Self::Error> {
    Ok(self.solve(initial))
  }
}

impl SvjCalibrator {
  /// Create a calibrator for quotes that share one maturity `tau`.
  pub fn new(
    params: Option<SvjParams>,
    c_market: Array1<f64>,
    s: Array1<f64>,
    k: Array1<f64>,
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
    params: Option<SvjParams>,
    slices: &[crate::calibration::levy::MarketSlice],
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
        flat_t.push(slice.tau);
        flat_s.push(s);
      }
    }

    Self {
      params,
      c_market: Array1::from_vec(flat_prices),
      s: Array1::from_vec(flat_s),
      k: Array1::from_vec(flat_strikes),
      r,
      q,
      flat_t,
      option_type,
      record_history,
      loss_metrics: &LossMetric::ALL,
      calibration_history: Rc::new(RefCell::new(Vec::new())),
    }
  }

  fn solve(&self, initial_params: Option<SvjParams>) -> SvjCalibrationResult {
    let mut problem = self.clone();
    if let Some(p) = initial_params {
      problem.params = Some(p.projected());
    }
    problem.ensure_initial_guess();

    let (result, report) = minimize(problem, LmOptions::default());

    let p = result.effective_params();
    let c_model = result.compute_model_prices_for(&p);
    let loss = CalibrationLossScore::compute_selected(
      result.c_market.as_standard_layout().as_slice().unwrap(),
      c_model.as_slice().unwrap(),
      result.loss_metrics,
    );

    SvjCalibrationResult {
      v0: p.v0,
      kappa: p.kappa,
      theta: p.theta,
      sigma_v: p.sigma_v,
      rho: p.rho,
      lambda: p.lambda,
      mu_j: p.mu_j,
      sigma_j: p.sigma_j,
      loss,
      converged: report.converged,
    }
  }

  pub fn set_initial_guess(&mut self, params: SvjParams) {
    self.params = Some(params.projected());
  }

  pub fn set_record_history(&mut self, record: bool) {
    self.record_history = record;
  }

  pub fn history(&self) -> Vec<CalibrationHistory<SvjParams>> {
    self.calibration_history.borrow().clone()
  }

  fn fallback_params() -> SvjParams {
    SvjParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma_v: 0.5,
      rho: -0.5,
      lambda: 0.5,
      mu_j: -0.05,
      sigma_j: 0.1,
    }
    .projected()
  }

  fn ensure_initial_guess(&mut self) {
    if self.params.is_none() {
      self.params = Some(Self::fallback_params());
    }
  }

  pub(super) fn effective_params(&self) -> SvjParams {
    if let Some(p) = &self.params {
      return (*p).projected();
    }
    Self::fallback_params()
  }
}
