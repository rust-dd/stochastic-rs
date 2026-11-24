use ndarray::Array1;

use super::loss::RoughHestonProblem;
use super::params::bounds;
use super::params::folded;
use super::params::held;
use super::params::room_in;
use crate::OptionQuote;
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationOptions;
use crate::calibration::least_squares::minimize_in_box;
use crate::calibration::quote::assert_spot_and_rates;
use crate::calibration::quote::validate_quotes;
use crate::pricing::fourier::RoughHestonCf;
use crate::pricing::fourier::RoughHestonParams;

/// Rough Heston fit to vanilla quotes by Levenberg–Marquardt in a box, each price residual
/// weighted by the square root of its quote's inverse Black vega.
#[derive(Clone, Debug)]
pub struct RoughHestonCalibrator {
  pub(super) s: f64,
  pub(super) r: f64,
  pub(super) q: f64,
  /// The weights are read off these prices when a fit starts.
  pub quotes: Vec<OptionQuote>,
  /// `None` starts from (0.04, 1, 0.04, 0.3, −0.65, 0.1); a start outside the box is reflected
  /// into it, not clamped.
  pub start: Option<RoughHestonParams>,
  /// The characteristic function the fit prices and converts by. The approximant's validated `ν`
  /// and `ρ` range cuts the box; a fit left on that cut reports it, on a model bound converges.
  pub cf: RoughHestonCf,
  /// Levenberg–Marquardt knobs; `regularization` pulls `(V₀, κ, θ, ν, ρ, H)`.
  pub options: CalibrationOptions,
}

impl RoughHestonCalibrator {
  /// Panics unless `s` is finite and positive and `r`, `q` finite; `calibrate` checks the quotes.
  pub fn new(s: f64, r: f64, q: f64, quotes: Vec<OptionQuote>) -> Self {
    assert_spot_and_rates(s, r, q);
    Self {
      s,
      r,
      q,
      quotes,
      start: None,
      cf: RoughHestonCf::default(),
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

  /// Panics unless `start` passes [`RoughHestonParams::new`].
  pub fn with_start(mut self, start: RoughHestonParams) -> Self {
    self.start = Some(start.checked());
    self
  }

  /// Panics unless `cf` passes `RoughHeston::new`'s checks. The Padé default is smooth and fast;
  /// `Adams` costs `O(steps²)` a frequency.
  pub fn with_cf(mut self, cf: RoughHestonCf) -> Self {
    if let Some(reason) = cf.violation() {
      panic!("{reason}");
    }
    self.cf = cf;
    self
  }

  pub fn with_options(mut self, options: CalibrationOptions) -> Self {
    self.options = options;
    self
  }

  pub(super) fn fallback_params() -> RoughHestonParams {
    RoughHestonParams {
      v0: 0.04,
      kappa: 1.0,
      theta: 0.04,
      nu: 0.3,
      rho: -0.65,
      hurst: 0.1,
    }
  }
}

impl crate::traits::Calibrator for RoughHestonCalibrator {
  type Params = RoughHestonParams;
  type Output = Calibrated<RoughHestonParams, RoughHestonCf>;
  type Error = QuantError;

  /// A start replaces `self.start`. A fit left on its selector's validity bound is not converged
  /// and says so; `InvalidInput` for a start or `cf` off its domain, `InvalidQuote`, `Optimizer`.
  fn calibrate(&self, initial: Option<RoughHestonParams>) -> Result<Self::Output, QuantError> {
    validate_quotes(&self.quotes)?;
    let start = initial.or(self.start).unwrap_or_else(Self::fallback_params);
    if let Some(reason) = start.violation() {
      return Err(QuantError::invalid("start", reason));
    }
    if let Some(reason) = self.cf.violation() {
      return Err(QuantError::invalid("cf", reason));
    }
    let lm = self.options.lm(true, 6)?;
    let bounds = bounds(&self.cf);
    let start = folded(Array1::from(start).view(), &bounds);
    let problem = RoughHestonProblem::new(self, start, bounds)?;
    let (fitted, report) = minimize_in_box(problem, lm, room_in(bounds));
    let model = fitted.model_prices(&fitted.params);
    let fit = report.calibrated(
      fitted.params,
      &fitted.market,
      &model,
      self.options.loss_metrics,
      fitted.trail.into_entries(),
    )?;
    let fit = match held(fit.params, &self.cf) {
      Some(name) => fit.at_bound(name),
      None => fit,
    };
    Ok(fit.with_diagnostics(self.cf))
  }
}
