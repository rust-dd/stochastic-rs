use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

use super::loss::default_params;
use super::loss::domain_violation;
use super::loss::fold_params;
use super::loss::fourier_option_price;
use super::loss::param_bounds;
use super::loss::param_count;
use super::loss::project_params;
use super::loss::room_in_box;
use super::types::LevyModelType;
use super::types::LevyParams;
use crate::OptionQuote;
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationOptions;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::calibration::least_squares::Trail;
use crate::calibration::least_squares::minimize_in_box;
use crate::calibration::quote::assert_spot_and_rates;
use crate::calibration::quote::validate_quotes;
use crate::calibration::scale_columns;
use crate::calibration::stencil::derivative;
use crate::calibration::stencil::step;
use crate::pricing::cf_quadrature::Rule;
use crate::pricing::cf_quadrature::Walk;

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

  /// Panics unless `start` is this model's, of its parameter count, finite and inside the model's
  /// `E[e^X] < ∞` as given; the fit starts from it clamped to bounds, then `0.01` inside the domain.
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
  } else if let Some((i, value)) = start
    .values
    .iter()
    .enumerate()
    .find(|(_, v)| !v.is_finite())
  {
    Some(format!(
      "start.values[{i}] must satisfy `start.values[{i}].is_finite()`, got start.values[{i}] = {value:?}"
    ))
  } else {
    domain_violation(model_type, &start.values)
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

  /// A start replaces `self.start`; `InvalidInput` unless it is this model's, finite and inside
  /// `E[e^X] < ∞`, `DimensionMismatch` for a wrong count, `Optimizer`, and the options' errors.
  fn calibrate(&self, initial: Option<LevyParams>) -> Result<Calibrated<LevyParams>, QuantError> {
    validate_quotes(&self.quotes)?;
    let start = initial.or_else(|| self.start.clone());
    if let Some(start) = &start {
      check_start(self.model_type, start)?;
    }
    let mut values = start.map_or_else(|| default_params(self.model_type), |start| start.values);
    project_params(self.model_type, &mut values);
    let model_type = self.model_type;
    let lm = self.options.lm(false, param_count(model_type))?;
    let rooms = move |x: ArrayView1<f64>| room_in_box(model_type, x);
    let (fitted, report) = minimize_in_box(LevyProblem::new(self, values), lm, rooms);
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
  pub(super) params: LevyParams,
  /// The coordinates folded into their bounds, before the model's domain moves one.
  folded: Vec<f64>,
  /// `dθⱼ/dxⱼ` of the bounds' reflection at the optimiser's raw coordinates.
  reflections: Array1<f64>,
  market: Vec<f64>,
  trail: Trail<LevyParams>,
}

impl<'a> LevyProblem<'a> {
  pub(super) fn new(calibrator: &'a LevyCalibrator, values: Vec<f64>) -> Self {
    Self {
      calibrator,
      folded: values.clone(),
      reflections: Array1::ones(values.len()),
      params: LevyParams {
        values,
        model_type: calibrator.model_type,
      },
      market: calibrator.quotes.iter().map(|quote| quote.price).collect(),
      trail: Trail::new(),
    }
  }

  fn price(&self, params: &[f64], quote: &OptionQuote, walk: &mut Walk<'_, Rule>) -> f64 {
    let LevyCalibrator {
      model_type,
      s,
      r,
      q,
      ..
    } = *self.calibrator;
    let (k, tau) = (quote.strike, quote.tau);
    fourier_option_price(model_type, params, s, k, r, q, tau, quote.option_type, walk)
  }

  fn compute_model_prices(&self) -> Vec<f64> {
    let quotes = self.calibrator.quotes.iter();
    quotes
      .map(|quote| self.price(&self.params.values, quote, &mut Walk::Adaptive))
      .collect()
  }

  /// `∂r/∂θ` at the folded point, the domain's projection inside each stencil and every price on
  /// the rule the point's own price walked; `scale` times the shared step.
  pub(super) fn difference_jacobian(&self, scale: f64) -> Array2<f64> {
    let model_type = self.calibrator.model_type;
    let quotes = &self.calibrator.quotes;
    let rules = quotes.iter().map(|quote| {
      let mut rules = Vec::new();
      self.price(&self.params.values, quote, &mut Walk::Record(&mut rules));
      rules
    });
    let rules = rules.collect::<Vec<_>>();
    let mut jacobian = Array2::zeros((quotes.len(), self.folded.len()));
    let columns = jacobian
      .columns_mut()
      .into_iter()
      .zip(param_bounds(model_type));
    for (j, (mut column, bounds)) in columns.enumerate() {
      let x = self.folded[j];
      column.assign(&derivative(x, scale * step(x), bounds, |value| {
        let mut moved = self.folded.clone();
        moved[j] = value;
        project_params(model_type, &mut moved);
        let replay = self.market.iter().zip(quotes).zip(&rules);
        Array1::from_iter(replay.map(|((market, quote), rules)| {
          market - self.price(&moved, quote, &mut Walk::Replay(rules.iter()))
        }))
      }));
    }
    jacobian
  }
}

impl LeastSquaresProblem for LevyProblem<'_> {
  fn set_params(&mut self, params: ArrayView1<f64>) {
    let model_type = self.calibrator.model_type;
    (self.folded, self.reflections) = fold_params(model_type, params);
    let mut p = self.folded.clone();
    project_params(model_type, &mut p);
    self.params.values = p;
  }

  fn params(&self) -> Array1<f64> {
    Array1::from_vec(self.folded.clone())
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
    let jacobian = self.difference_jacobian(1.0);
    Some(scale_columns(jacobian, self.reflections.view()))
  }

  fn accept(&mut self) {
    self.trail.accept();
  }

  fn natural_params(&self) -> Vec<f64> {
    self.params.values.clone()
  }

  fn natural_jacobian(&self) -> Array2<f64> {
    Array2::from_diag(&self.reflections)
  }
}
