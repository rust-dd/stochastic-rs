use super::loss::HkdeProblem;
use crate::OptionQuote;
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationOptions;
use crate::calibration::least_squares::minimize;
use crate::calibration::quote::assert_spot_and_rates;
use crate::calibration::quote::validate_quotes;
use crate::pricing::fourier::HkdeParams;

/// Heston–Kou fit to vanilla quotes by Levenberg–Marquardt, each price residual weighted by the
/// square root of its quote's inverse Black vega (Agazzotti et al. 2025, eq. 13).
#[derive(Clone, Debug)]
pub struct HkdeCalibrator {
  pub(super) s: f64,
  pub(super) r: f64,
  pub(super) q: f64,
  /// The weights are read off these prices when a fit starts.
  pub quotes: Vec<OptionQuote>,
  /// `None` starts from (0.04, 1.5, 0.04, 0.3, −0.5, 0.5, 0.4, 10, 5), projected.
  pub start: Option<HkdeParams>,
  /// Levenberg–Marquardt knobs; `regularization` pulls `(v0, κ, θ, σ_v, ρ, λ, p, η₁, η₂)`.
  pub options: CalibrationOptions,
}

impl HkdeCalibrator {
  /// Panics unless `s` is finite and positive and `r`, `q` finite; `calibrate` checks the quotes.
  pub fn new(s: f64, r: f64, q: f64, quotes: Vec<OptionQuote>) -> Self {
    assert_spot_and_rates(s, r, q);
    Self {
      s,
      r,
      q,
      quotes,
      start: None,
      options: CalibrationOptions::default(),
    }
  }

  pub fn s(&self) -> f64 {
    self.s
  }

  pub fn r(&self) -> f64 {
    self.r
  }

  pub fn q(&self) -> f64 {
    self.q
  }

  /// Panics unless `start` passes `HkdeFourier::new`'s checks; the fit starts from its projection
  /// onto the calibration box.
  pub fn with_start(mut self, start: HkdeParams) -> Self {
    if let Some(reason) = start.violation() {
      panic!("{reason}");
    }
    self.start = Some(start);
    self
  }

  pub fn with_options(mut self, options: CalibrationOptions) -> Self {
    self.options = options;
    self
  }

  pub(super) fn fallback_params() -> HkdeParams {
    HkdeParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.5,
      lambda: 0.5,
      p_up: 0.4,
      eta1: 10.0,
      eta2: 5.0,
    }
    .projected()
  }
}

impl crate::traits::Calibrator for HkdeCalibrator {
  type Params = HkdeParams;
  type Output = Calibrated<HkdeParams>;
  type Error = QuantError;

  /// A start replaces `self.start`; `InvalidInput` unless it passes `HkdeFourier::new`,
  /// `InvalidQuote` for a quote no vol at `r − q` weights, `Optimizer`, the options' errors.
  fn calibrate(&self, initial: Option<HkdeParams>) -> Result<Calibrated<HkdeParams>, QuantError> {
    validate_quotes(&self.quotes)?;
    let start = initial.or(self.start).unwrap_or_else(Self::fallback_params);
    if let Some(reason) = start.violation() {
      return Err(QuantError::invalid("start", reason));
    }
    let lm = self.options.lm(true, 9)?;
    let (fitted, report) = minimize(HkdeProblem::new(self, start)?, lm);
    let params = fitted.effective_params();
    let model = fitted.compute_model_prices_for(&params);
    report.calibrated(
      params,
      &fitted.market,
      &model,
      self.options.loss_metrics,
      fitted.trail.into_entries(),
    )
  }
}
