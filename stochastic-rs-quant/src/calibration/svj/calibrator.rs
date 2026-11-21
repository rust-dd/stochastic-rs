use super::SvjParams;
use super::loss::SvjProblem;
use super::params::outward;
use crate::OptionQuote;
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationOptions;
use crate::calibration::least_squares::minimize_in_box;
use crate::calibration::quote::assert_spot_and_rates;
use crate::calibration::quote::validate_quotes;

pub(super) const KAPPA_MIN: f64 = 1e-3;

pub(super) const THETA_MIN: f64 = 1e-8;

pub(super) const SIGMA_V_MIN: f64 = 1e-8;

/// Bates (SVJ) fit to vanilla quotes by Levenberg–Marquardt on a central-difference Jacobian.
/// Levenberg (1944), DOI 10.1090/qam/10666; Marquardt (1963), DOI 10.1137/0111030.
#[derive(Clone, Debug)]
pub struct SvjCalibrator {
  pub(super) s: f64,
  pub(super) r: f64,
  pub(super) q: f64,
  pub quotes: Vec<OptionQuote>,
  /// `None` starts from (0.04, 1.5, 0.04, 0.5, −0.5, 0.5, −0.05, 0.1), projected.
  pub start: Option<SvjParams>,
  /// Levenberg–Marquardt knobs; `regularization` pulls `(v0, κ, θ, σ_v, ρ, λ, μ_J, σ_J)`.
  pub options: CalibrationOptions,
}

impl SvjCalibrator {
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

  /// Panics unless `start` passes `BatesFourier::new`'s checks; the fit starts from its projection
  /// onto the calibration box.
  pub fn with_start(mut self, start: SvjParams) -> Self {
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

  pub(super) fn fallback_params() -> SvjParams {
    SvjParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.5,
      rho: -0.5,
      lambda: 0.5,
      mu_j: -0.05,
      sigma_j: 0.1,
    }
    .projected()
  }
}

impl crate::traits::Calibrator for SvjCalibrator {
  type Params = SvjParams;
  type Output = Calibrated<SvjParams>;
  type Error = QuantError;

  /// A start replaces `self.start`; `InvalidInput` unless the one used passes `BatesFourier::new`,
  /// `Optimizer` when the fit cannot evaluate it, and the `CalibrationOptions` errors.
  fn calibrate(&self, initial: Option<SvjParams>) -> Result<Calibrated<SvjParams>, QuantError> {
    validate_quotes(&self.quotes)?;
    let start = initial.or(self.start).unwrap_or_else(Self::fallback_params);
    if let Some(reason) = start.violation() {
      return Err(QuantError::invalid("start", reason));
    }
    let lm = self.options.lm(true, 8)?;
    let (fitted, report) = minimize_in_box(SvjProblem::new(self, start), lm, outward);
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
