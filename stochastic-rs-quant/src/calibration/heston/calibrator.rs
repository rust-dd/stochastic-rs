use std::cell::RefCell;
use std::rc::Rc;

use anyhow::Result;
use anyhow::bail;
use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DVector;
use ndarray::Array1;
use stochastic_rs_stats::heston_nml_cekf::HestonNmleCekfConfig;

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
/// - Levenberg (1944), <https://doi.org/10.1090/qam/10666>
/// - Marquardt (1963), <https://doi.org/10.1137/0111030>
/// - Heston model (1993), <https://doi.org/10.1093/rfs/6.2.327>
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
  /// Positive row weights applied to price residuals and their Jacobian.
  pub residual_weights: DVector<f64>,
  /// Optional: time series for MLE-based initial guess
  pub mle_s: Option<Array1<f64>>,
  pub mle_v: Option<Array1<f64>>,
  pub mle_r: Option<f64>,
  /// Seed method for the MLE-based initial guess.
  pub mle_seed_method: HestonMleSeedMethod,
  /// Optional explicit sampling step used by MLE seed estimators.
  pub mle_delta: Option<f64>,
  /// Optional config for NMLE-CEKF seed when `mle_seed_method = NmleCekf`.
  pub nmle_cekf_config: Option<HestonNmleCekfConfig>,
  /// If true, record per-iteration calibration history.
  pub record_history: bool,
  /// Which loss metrics to compute when recording history.
  pub loss_metrics: &'static [LossMetric],
  /// Jacobian/method choice for calibration.
  pub jacobian_method: HestonJacobianMethod,
  /// Optional common LM tolerance for research workloads.
  pub optimizer_tolerance: Option<f64>,
  /// Maximum residual-evaluation budget per fitted parameter, as defined by LM.
  pub optimizer_patience: usize,
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
      residual_weights: DVector::from_element(n, 1.0),
      mle_s,
      mle_v,
      mle_r,
      mle_seed_method: HestonMleSeedMethod::default(),
      mle_delta: None,
      nmle_cekf_config: None,
      record_history,
      loss_metrics: &LossMetric::ALL,
      jacobian_method: HestonJacobianMethod::default(),
      optimizer_tolerance: None,
      optimizer_patience: 100,
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
        flat_t.push(slice.tau);
        flat_s.push(s);
      }
    }
    let quote_count = flat_prices.len();

    Self {
      params,
      c_market: DVector::from_vec(flat_prices),
      s: DVector::from_vec(flat_s),
      k: DVector::from_vec(flat_strikes),
      r,
      q,
      flat_t,
      option_type,
      residual_weights: DVector::from_element(quote_count, 1.0),
      mle_s: None,
      mle_v: None,
      mle_r: None,
      mle_seed_method: HestonMleSeedMethod::default(),
      mle_delta: None,
      nmle_cekf_config: None,
      record_history,
      loss_metrics: &LossMetric::ALL,
      jacobian_method: HestonJacobianMethod::default(),
      optimizer_tolerance: None,
      optimizer_patience: 100,
      calibration_history: Rc::new(RefCell::new(Vec::new())),
    }
  }
}

impl HestonCalibrator {
  pub(super) fn solve(&self) -> HestonCalibrationResult {
    let mut problem = self.clone();
    problem.ensure_initial_guess();

    let mut optimizer = LevenbergMarquardt::new().with_patience(self.optimizer_patience);
    if let Some(tolerance) = self.optimizer_tolerance {
      optimizer = optimizer.with_tol(tolerance);
    }
    let (result, report) = optimizer.minimize(problem);
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

  /// Set positive residual weights, normalized to unit root-mean-square.
  ///
  /// Inverse Black-Scholes vega weights make the least-squares objective a
  /// first-order approximation to implied-volatility error.
  pub fn set_residual_weights(&mut self, weights: impl Into<DVector<f64>>) -> Result<()> {
    let mut weights = weights.into();
    if weights.len() != self.c_market.len() {
      bail!(
        "Heston residual weight count {} does not match quote count {}",
        weights.len(),
        self.c_market.len()
      );
    }
    if weights.is_empty()
      || weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
      bail!("Heston residual weights must be finite and positive");
    }
    let root_mean_square =
      (weights.iter().map(|weight| weight * weight).sum::<f64>() / weights.len() as f64).sqrt();
    if !root_mean_square.is_finite() || root_mean_square <= 0.0 {
      bail!("Heston residual weight normalization is invalid");
    }
    weights /= root_mean_square;
    self.residual_weights = weights;
    Ok(())
  }

  /// Enable or disable recording of per-iteration calibration history.
  pub fn set_record_history(&mut self, record: bool) {
    self.record_history = record;
  }

  pub fn set_jacobian_method(&mut self, method: HestonJacobianMethod) {
    self.jacobian_method = method;
  }

  /// Set a positive common LM convergence tolerance.
  pub fn set_optimizer_tolerance(&mut self, tolerance: Option<f64>) -> Result<()> {
    if tolerance.is_some_and(|value| !value.is_finite() || value <= 0.0) {
      bail!("Heston optimizer tolerance must be finite and positive");
    }
    self.optimizer_tolerance = tolerance;
    Ok(())
  }

  /// Set the LM evaluation-budget multiplier.
  pub fn set_optimizer_patience(&mut self, patience: usize) -> Result<()> {
    if patience == 0 {
      bail!("Heston optimizer patience must be positive");
    }
    self.optimizer_patience = patience;
    Ok(())
  }

  pub fn set_mle_seed_method(&mut self, method: HestonMleSeedMethod) {
    self.mle_seed_method = method;
  }

  pub fn set_mle_delta(&mut self, delta: Option<f64>) {
    self.mle_delta = delta;
  }

  pub fn set_nmle_cekf_config(&mut self, cfg: HestonNmleCekfConfig) {
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
