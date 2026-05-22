use std::cell::RefCell;
use std::rc::Rc;

use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DVector;
use ndarray::Array1;
use stochastic_rs_stats::heston_nml_cekf::HestonNMLECEKFConfig;

use super::params::HestonJacobianMethod;
use super::params::HestonMleSeedMethod;
use super::params::HestonParams;
use super::result::HestonCalibrationResult;
use crate::CalibrationLossScore;
use crate::LossMetric;
use crate::OptionType;
use crate::calibration::CalibrationHistory;

#[derive(Clone)]
/// Heston least-squares calibrator using Levenberg-Marquardt iterations.
///
/// Source:
/// - Levenberg (1944), https://doi.org/10.1090/qam/10666
/// - Marquardt (1963), https://doi.org/10.1137/0111030
/// - Heston model (1993), https://doi.org/10.1093/rfs/6.2.327
pub struct HestonCalibrator {
  /// Params to calibrate (v0, kappa, theta, sigma, rho).
  /// If None, an initial guess will be inferred using heston_mle (requires mle_* fields).
  pub params: Option<HestonParams>,
  /// Option prices from the market (flattened across all maturities).
  pub c_market: DVector<f64>,
  /// Underlying spot per quote (allows small variations per strike/maturity bucket).
  pub s: DVector<f64>,
  /// Strikes per quote (flattened).
  pub k: DVector<f64>,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: Option<f64>,
  /// Time to maturity per quote (flattened). Supports multi-maturity joint calibration.
  pub flat_t: Vec<f64>,
  /// Option type of the quotes.
  pub option_type: OptionType,
  /// Optional: time series for MLE-based initial guess
  pub mle_s: Option<Array1<f64>>,
  pub mle_v: Option<Array1<f64>>,
  pub mle_r: Option<f64>,
  /// Seed method for the MLE-based initial guess.
  pub mle_seed_method: HestonMleSeedMethod,
  /// Optional explicit sampling step used by MLE seed estimators.
  pub mle_delta: Option<f64>,
  /// Optional config for NMLE-CEKF seed when `mle_seed_method = NmleCekf`.
  pub nmle_cekf_config: Option<HestonNMLECEKFConfig>,
  /// If true, record per-iteration calibration history.
  pub record_history: bool,
  /// Which loss metrics to compute when recording history.
  pub loss_metrics: &'static [LossMetric],
  /// Jacobian/method choice for calibration.
  pub jacobian_method: HestonJacobianMethod,
  /// History of iterations (residuals, params, loss metrics).
  pub(super) calibration_history: Rc<RefCell<Vec<CalibrationHistory<HestonParams>>>>,
}

impl HestonCalibrator {
  /// Create a calibrator for a single maturity slice (backwards compatible).
  pub fn new(
    params: Option<HestonParams>,
    c_market: DVector<f64>,
    s: DVector<f64>,
    k: DVector<f64>,
    r: f64,
    q: Option<f64>,
    tau: f64,
    option_type: OptionType,
    mle_s: Option<Array1<f64>>,
    mle_v: Option<Array1<f64>>,
    mle_r: Option<f64>,
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
      mle_s,
      mle_v,
      mle_r,
      mle_seed_method: HestonMleSeedMethod::default(),
      mle_delta: None,
      nmle_cekf_config: None,
      record_history,
      loss_metrics: &LossMetric::ALL,
      jacobian_method: HestonJacobianMethod::default(),
      calibration_history: Rc::new(RefCell::new(Vec::new())),
    }
  }

  /// Create a calibrator from multiple maturity slices for joint surface calibration.
  pub fn from_slices(
    params: Option<HestonParams>,
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
      mle_s: None,
      mle_v: None,
      mle_r: None,
      mle_seed_method: HestonMleSeedMethod::default(),
      mle_delta: None,
      nmle_cekf_config: None,
      record_history,
      loss_metrics: &LossMetric::ALL,
      jacobian_method: HestonJacobianMethod::default(),
      calibration_history: Rc::new(RefCell::new(Vec::new())),
    }
  }
}

impl HestonCalibrator {
  pub(super) fn solve(&self) -> HestonCalibrationResult {
    let mut problem = self.clone();
    problem.ensure_initial_guess();

    let (result, report) = LevenbergMarquardt::new().minimize(problem);
    let converged = report.termination.was_successful();
    let params = result.effective_params();

    let c_model = result.compute_model_prices_for_numeric(&params);
    let loss = CalibrationLossScore::compute_selected(
      result.c_market.as_slice(),
      c_model.as_slice(),
      result.loss_metrics,
    );

    HestonCalibrationResult {
      params,
      loss,
      converged,
    }
  }

  pub fn set_initial_guess(&mut self, params: HestonParams) {
    self.params = Some(params.projected());
  }

  /// Enable or disable recording of per-iteration calibration history.
  pub fn set_record_history(&mut self, record: bool) {
    self.record_history = record;
  }

  pub fn set_jacobian_method(&mut self, method: HestonJacobianMethod) {
    self.jacobian_method = method;
  }

  pub fn set_mle_seed_method(&mut self, method: HestonMleSeedMethod) {
    self.mle_seed_method = method;
  }

  pub fn set_mle_delta(&mut self, delta: Option<f64>) {
    self.mle_delta = delta;
  }

  pub fn set_nmle_cekf_config(&mut self, cfg: HestonNMLECEKFConfig) {
    self.nmle_cekf_config = Some(cfg);
  }

  /// Retrieve the collected calibration history.
  pub fn history(&self) -> Vec<CalibrationHistory<HestonParams>> {
    self.calibration_history.borrow().clone()
  }
}

impl crate::traits::Calibrator for HestonCalibrator {
  type InitialGuess = HestonParams;
  type Params = HestonParams;
  type Output = HestonCalibrationResult;
  type Error = anyhow::Error;

  fn calibrate(&self, initial: Option<Self::InitialGuess>) -> Result<Self::Output, Self::Error> {
    let mut this = self.clone();
    if let Some(p) = initial {
      this.set_initial_guess(p);
    }
    Ok(this.solve())
  }
}
