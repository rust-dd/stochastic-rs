use super::loss::DoubleHestonProblem;
use super::params::DoubleHestonParams;
use super::params::room_in_box;
use crate::OptionQuote;
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationOptions;
use crate::calibration::least_squares::minimize_in_box;
use crate::calibration::quote::assert_spot_and_rates;
use crate::calibration::quote::validate_quotes;
use crate::pricing::heston::HestonParams;

/// Double-Heston fit to vanilla quotes by Levenberg–Marquardt on a central-difference Jacobian.
#[derive(Clone, Debug)]
pub struct DoubleHestonCalibrator {
  pub(super) s: f64,
  pub(super) r: f64,
  pub(super) q: f64,
  pub quotes: Vec<OptionQuote>,
  /// `None` starts from a fast factor (0.02, 3, 0.02, 0.3, −0.6) and a slow one
  /// (0.02, 0.5, 0.02, 0.15, −0.3), projected.
  pub start: Option<DoubleHestonParams>,
  /// Levenberg–Marquardt knobs; `regularization` pulls the first factor's `(v0, κ, θ, σ, ρ)`, then
  /// the second's.
  pub options: CalibrationOptions,
}

impl DoubleHestonCalibrator {
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

  /// Panics, naming the factor, unless both pass [`HestonParams::new`]; the fit starts from the
  /// start's projection onto the calibration box.
  pub fn with_start(mut self, start: DoubleHestonParams) -> Self {
    if let Some(reason) = start_violation(&start) {
      panic!("{reason}");
    }
    self.start = Some(start);
    self
  }

  pub fn with_options(mut self, options: CalibrationOptions) -> Self {
    self.options = options;
    self
  }

  pub(super) fn fallback_params() -> DoubleHestonParams {
    DoubleHestonParams {
      first: HestonParams {
        v0: 0.02,
        kappa: 3.0,
        theta: 0.02,
        sigma: 0.3,
        rho: -0.6,
      },
      second: HestonParams {
        v0: 0.02,
        kappa: 0.5,
        theta: 0.02,
        sigma: 0.15,
        rho: -0.3,
      },
    }
    .projected()
  }
}

/// The first factor outside the Heston domain, named, in `HestonParams::new`'s words.
fn start_violation(start: &DoubleHestonParams) -> Option<String> {
  [("first", start.first), ("second", start.second)]
    .into_iter()
    .find_map(|(factor, params)| {
      params
        .violation()
        .map(|reason| format!("{factor} factor: {reason}"))
    })
}

impl crate::traits::Calibrator for DoubleHestonCalibrator {
  type Params = DoubleHestonParams;
  type Output = Calibrated<DoubleHestonParams>;
  type Error = QuantError;

  /// A start replaces `self.start`; `InvalidInput` unless its factors pass `HestonParams::new`,
  /// `Optimizer` when the fit cannot evaluate it, and the `CalibrationOptions` errors.
  fn calibrate(
    &self,
    initial: Option<DoubleHestonParams>,
  ) -> Result<Calibrated<DoubleHestonParams>, QuantError> {
    validate_quotes(&self.quotes)?;
    let start = initial.or(self.start).unwrap_or_else(Self::fallback_params);
    if let Some(reason) = start_violation(&start) {
      return Err(QuantError::invalid("start", reason));
    }
    let lm = self.options.lm(true, 10)?;
    let (fitted, report) = minimize_in_box(DoubleHestonProblem::new(self, start), lm, room_in_box);
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
