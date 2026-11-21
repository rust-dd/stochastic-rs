use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

use super::EPS;
use super::loss::default_params;
use super::loss::fourier_option_price;
use super::loss::param_bounds;
use super::loss::param_count;
use super::loss::project_params;
use super::types::LevyModelType;
use super::types::LevyParams;
use crate::OptionQuote;
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationOptions;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::calibration::least_squares::Trail;
use crate::calibration::least_squares::minimize;
use crate::calibration::quote::assert_spot_and_rates;
use crate::calibration::quote::validate_quotes;

/// Lévy-model fit to vanilla quotes by Levenberg–Marquardt on a central-difference Jacobian.
/// Levenberg (1944), DOI 10.1090/qam/10666; Marquardt (1963), DOI 10.1137/0111030.
#[derive(Clone, Debug)]
pub struct LevyCalibrator {
  pub model_type: LevyModelType,
  s: f64,
  r: f64,
  q: f64,
  pub quotes: Vec<OptionQuote>,
  /// `None` starts from the model's default parameters.
  pub start: Option<LevyParams>,
  /// Levenberg–Marquardt knobs; `regularization` pulls `LevyParams::values` in the model's order.
  pub options: CalibrationOptions,
}

impl LevyCalibrator {
  /// Panics unless `s` is finite and positive and `r`, `q` finite; `calibrate` checks the quotes.
  pub fn new(model_type: LevyModelType, s: f64, r: f64, q: f64, quotes: Vec<OptionQuote>) -> Self {
    assert_spot_and_rates(s, r, q);
    Self {
      model_type,
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

  /// Panics unless `start.model_type == self.model_type`, `start.values.len()` is the model's
  /// parameter count and every value is finite; the fit starts from the values clamped to bounds.
  pub fn with_start(mut self, start: LevyParams) -> Self {
    if let Some(reason) = start_violation(self.model_type, &start) {
      panic!("{reason}");
    }
    self.start = Some(start);
    self
  }

  pub fn with_options(mut self, options: CalibrationOptions) -> Self {
    self.options = options;
    self
  }
}

/// The first way `start` does not fit `model_type`, in a constructor panic's words.
fn start_violation(model_type: LevyModelType, start: &LevyParams) -> Option<String> {
  let count = param_count(model_type);
  if start.model_type != model_type {
    Some(format!(
      "start.model_type must satisfy `start.model_type == {model_type:?}`, got start.model_type = {:?}",
      start.model_type
    ))
  } else if start.values.len() != count {
    Some(format!(
      "start.values must satisfy `start.values.len() == {count}`, got start.values.len() = {}",
      start.values.len()
    ))
  } else {
    let (i, value) = start
      .values
      .iter()
      .enumerate()
      .find(|(_, v)| !v.is_finite())?;
    Some(format!(
      "start.values[{i}] must satisfy `start.values[{i}].is_finite()`, got start.values[{i}] = {value:?}"
    ))
  }
}

/// `start_violation` as an `Err`, with a count other than the model's a `DimensionMismatch`.
fn check_start(model_type: LevyModelType, start: &LevyParams) -> Result<(), QuantError> {
  let expected = param_count(model_type);
  if start.model_type == model_type && start.values.len() != expected {
    return Err(QuantError::DimensionMismatch {
      what: "start values",
      expected,
      got: start.values.len(),
    });
  }
  match start_violation(model_type, start) {
    Some(reason) => Err(QuantError::invalid("start", reason)),
    None => Ok(()),
  }
}

impl crate::traits::Calibrator for LevyCalibrator {
  type Params = LevyParams;
  type Output = Calibrated<LevyParams>;
  type Error = QuantError;

  /// A start replaces `self.start`; `InvalidInput` unless it is this model's and finite,
  /// `DimensionMismatch` for a wrong count, `Optimizer`, and the `CalibrationOptions` errors.
  fn calibrate(&self, initial: Option<LevyParams>) -> Result<Calibrated<LevyParams>, QuantError> {
    validate_quotes(&self.quotes)?;
    let start = initial.or_else(|| self.start.clone());
    if let Some(start) = &start {
      check_start(self.model_type, start)?;
    }
    let mut values = start.map_or_else(|| default_params(self.model_type), |start| start.values);
    project_params(self.model_type, &mut values);
    let lm = self.options.lm(false, param_count(self.model_type))?;
    let (fitted, report) = minimize(LevyProblem::new(self, values), lm);
    let model = fitted.compute_model_prices();
    report.calibrated(
      fitted.params,
      &fitted.market,
      &model,
      self.options.loss_metrics,
      fitted.trail.into_entries(),
    )
  }
}

pub(super) struct LevyProblem<'a> {
  calibrator: &'a LevyCalibrator,
  params: LevyParams,
  market: Vec<f64>,
  trail: Trail<LevyParams>,
}

impl<'a> LevyProblem<'a> {
  pub(super) fn new(calibrator: &'a LevyCalibrator, values: Vec<f64>) -> Self {
    Self {
      calibrator,
      params: LevyParams {
        values,
        model_type: calibrator.model_type,
      },
      market: calibrator.quotes.iter().map(|quote| quote.price).collect(),
      trail: Trail::new(),
    }
  }

  fn price(&self, params: &[f64], quote: &OptionQuote) -> f64 {
    let LevyCalibrator {
      model_type,
      s,
      r,
      q,
      ..
    } = *self.calibrator;
    let (k, tau) = (quote.strike, quote.tau);
    fourier_option_price(model_type, params, s, k, r, q, tau, quote.option_type)
  }

  fn compute_model_prices(&self) -> Vec<f64> {
    let quotes = self.calibrator.quotes.iter();
    quotes
      .map(|quote| self.price(&self.params.values, quote))
      .collect()
  }

  fn effective_params(&self) -> Vec<f64> {
    let mut p = self.params.values.clone();
    project_params(self.calibrator.model_type, &mut p);
    p
  }

  fn numeric_jacobian(&self) -> Array2<f64> {
    let model_type = self.calibrator.model_type;
    let n = self.calibrator.quotes.len();
    let m = param_count(model_type);
    let bounds = param_bounds(model_type);
    let mut j_mat = Array2::zeros((n, m));
    let base = self.effective_params();

    for col in 0..m {
      let x = base[col];
      let h = 1e-5_f64.max(1e-3 * x.abs());

      let mut p_plus = base.clone();
      let mut p_minus = base.clone();
      p_plus[col] = (x + h).clamp(bounds[col].0, bounds[col].1);
      p_minus[col] = (x - h).clamp(bounds[col].0, bounds[col].1);
      project_params(model_type, &mut p_plus);
      project_params(model_type, &mut p_minus);

      let actual_h = p_plus[col] - p_minus[col];
      if actual_h.abs() < EPS {
        continue;
      }

      for (i, quote) in self.calibrator.quotes.iter().enumerate() {
        let f_plus = self.price(&p_plus, quote);
        let f_minus = self.price(&p_minus, quote);
        // The residual is market − model, so its derivative is −∂model.
        j_mat[(i, col)] = -(f_plus - f_minus) / actual_h;
      }
    }

    j_mat
  }
}

impl LeastSquaresProblem for LevyProblem<'_> {
  fn set_params(&mut self, params: ArrayView1<f64>) {
    let mut p: Vec<f64> = params.to_vec();
    project_params(self.calibrator.model_type, &mut p);
    self.params.values = p;
  }

  fn params(&self) -> Array1<f64> {
    Array1::from_vec(self.effective_params())
  }

  fn residuals(&mut self) -> Option<Array1<f64>> {
    let c_model = self.compute_model_prices();
    let rows = self.market.iter().zip(&c_model);
    let residuals = Array1::from_iter(rows.map(|(market, model)| market - model));
    let options = &self.calibrator.options;
    self.trail.stage(
      options,
      &self.market,
      &c_model,
      &self.params,
      residuals.view(),
    );
    Some(residuals)
  }

  fn jacobian(&self) -> Option<Array2<f64>> {
    Some(self.numeric_jacobian())
  }

  fn accept(&mut self) {
    self.trail.accept();
  }
}
