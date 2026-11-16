use super::loss::DoubleHestonProblem;
use super::params::DoubleHestonParams;
use crate::OptionQuote;
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationOptions;
use crate::calibration::least_squares::minimize;
use crate::calibration::quote::assert_spot_and_rates;
use crate::calibration::quote::validate_quotes;
use crate::pricing::heston::HestonParams;

/// Double-Heston fit to vanilla quotes by Levenberg–Marquardt on a central-difference Jacobian.
#[derive(Clone, Debug)]
pub struct DoubleHestonCalibrator {
  pub s: f64,
  pub r: f64,
  pub q: f64,
  pub quotes: Vec<OptionQuote>,
  /// `None` starts from a fast factor (0.02, 3, 0.02, 0.3, −0.6) and a slow one
  /// (0.02, 0.5, 0.02, 0.15, −0.3), projected.
  pub start: Option<DoubleHestonParams>,
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

  /// Panics unless both factors pass [`HestonParams::new`]; the fit starts from the start's
  /// projection onto the calibration box.
  pub fn with_start(mut self, start: DoubleHestonParams) -> Self {
    self.start = Some(DoubleHestonParams {
      first: start.first.checked(),
      second: start.second.checked(),
    });
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

/// `InvalidInput { what: "start" }` naming the first factor outside the Heston domain.
fn check_start(start: &DoubleHestonParams) -> Result<(), QuantError> {
  for (factor, params) in [("first", start.first), ("second", start.second)] {
    if let Some(reason) = params.violation() {
      return Err(QuantError::invalid(
        "start",
        format!("{factor} factor: {reason}"),
      ));
    }
  }
  Ok(())
}

impl crate::traits::Calibrator for DoubleHestonCalibrator {
  type Params = DoubleHestonParams;
  type Output = Calibrated<DoubleHestonParams>;
  type Error = QuantError;

  /// A start replaces `self.start`; `InvalidInput` unless both of its factors pass
  /// `HestonParams::new`.
  fn calibrate(
    &self,
    initial: Option<DoubleHestonParams>,
  ) -> Result<Calibrated<DoubleHestonParams>, QuantError> {
    validate_quotes(&self.quotes)?;
    let start = initial.or(self.start).unwrap_or_else(Self::fallback_params);
    check_start(&start)?;
    let (fitted, report) = minimize(DoubleHestonProblem::new(self, start), self.options.lm(true));
    let params = fitted.effective_params();
    let model = fitted.compute_model_prices_for(&params);
    let market = self
      .quotes
      .iter()
      .map(|quote| quote.price)
      .collect::<Vec<_>>();
    Ok(report.calibrated(
      params,
      &market,
      &model,
      self.options.loss_metrics,
      fitted.history,
    ))
  }
}
