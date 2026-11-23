use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

use super::calibrator::RoughHestonCalibrator;
use super::params::BOX;
use super::params::folded;
use crate::OptionQuote;
use crate::OptionType;
use crate::QuantError;
use crate::calibration::clip_negative;
use crate::calibration::hkde::market_weight;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::calibration::least_squares::Trail;
use crate::calibration::periodic_slopes;
use crate::calibration::scale_columns;
use crate::calibration::stencil::derivative;
use crate::calibration::stencil::step;
use crate::pricing::cf_quadrature::Rule;
use crate::pricing::cf_quadrature::Walk;
use crate::pricing::fourier::LewisPricer;
use crate::pricing::fourier::RoughHeston;
use crate::pricing::fourier::RoughHestonParams;

pub(super) struct RoughHestonProblem<'a> {
  calibrator: &'a RoughHestonCalibrator,
  sqrt_weights: Vec<f64>,
  /// The folded point, inside `BOX`.
  pub(super) params: RoughHestonParams,
  /// `dθⱼ/dxⱼ` of the box's reflection at the optimiser's raw coordinates.
  reflections: Array1<f64>,
  pub(super) market: Vec<f64>,
  pub(super) trail: Trail<RoughHestonParams>,
}

impl<'a> RoughHestonProblem<'a> {
  /// `InvalidQuote` for the first quote the inverse-vega weight cannot use.
  pub(super) fn new(
    calibrator: &'a RoughHestonCalibrator,
    start: RoughHestonParams,
  ) -> Result<Self, QuantError> {
    let RoughHestonCalibrator { s, r, q, .. } = *calibrator;
    let weights = calibrator.quotes.iter().enumerate().map(|(index, quote)| {
      market_weight(s, r, q, quote)
        .map(f64::sqrt)
        .map_err(|reason| QuantError::InvalidQuote { index, reason })
    });
    Ok(Self {
      calibrator,
      sqrt_weights: weights.collect::<Result<Vec<_>, _>>()?,
      params: start,
      reflections: Array1::ones(6),
      market: calibrator.quotes.iter().map(|quote| quote.price).collect(),
      trail: Trail::negated(),
    })
  }

  /// The quote's Lewis price, a put by parity, as `ModelPricer::price_option` gives it.
  fn price(&self, model: &RoughHeston, quote: &OptionQuote, walk: &mut Walk<'_, Rule>) -> f64 {
    let RoughHestonCalibrator { s, r, q, .. } = *self.calibrator;
    let (k, tau) = (quote.strike, quote.tau);
    let call = LewisPricer::price_call_walk(model, s, k, r, q, tau, walk);
    clip_negative(match quote.option_type {
      OptionType::Call => call,
      OptionType::Put => call - s * (-q * tau).exp() + k * (-r * tau).exp(),
    })
  }

  pub(super) fn model_prices(&self, p: &RoughHestonParams) -> Vec<f64> {
    let model = RoughHeston::new(*p, self.calibrator.cf);
    let quotes = self.calibrator.quotes.iter();
    quotes
      .map(|quote| self.price(&model, quote, &mut Walk::Adaptive))
      .collect()
  }

  /// `√w · (model − market)`, the fit's residual; a trail reads its negation.
  pub(super) fn weighted_residuals_of(&self, model: &[f64]) -> Array1<f64> {
    let rows = self.market.iter().zip(model).zip(&self.sqrt_weights);
    Array1::from_iter(rows.map(|((market, model), w)| w * (model - market)))
  }

  /// `∂r/∂x` at the folded point, every price on the rules the point's own prices walked; `scale`
  /// times the shared step.
  pub(super) fn difference_jacobian(&self, scale: f64) -> Array2<f64> {
    let point = Array1::from(self.params);
    let quotes = &self.calibrator.quotes;
    let at = RoughHeston::new(self.params, self.calibrator.cf);
    let rules = quotes.iter().map(|quote| {
      let mut rules = Vec::new();
      self.price(&at, quote, &mut Walk::Record(&mut rules));
      rules
    });
    let rules = rules.collect::<Vec<_>>();
    let mut jacobian = Array2::zeros((quotes.len(), point.len()));
    for (j, mut column) in jacobian.columns_mut().into_iter().enumerate() {
      let x = point[j];
      column.assign(&derivative(x, scale * step(x), BOX[j], |value| {
        let mut moved = point.clone();
        moved[j] = value;
        let moved = RoughHeston::new(RoughHestonParams::from(moved), self.calibrator.cf);
        let replay = quotes.iter().zip(&rules);
        let model =
          replay.map(|(quote, rules)| self.price(&moved, quote, &mut Walk::Replay(rules.iter())));
        self.weighted_residuals_of(&model.collect::<Vec<_>>())
      }));
    }
    jacobian
  }
}

impl LeastSquaresProblem for RoughHestonProblem<'_> {
  fn set_params(&mut self, params: ArrayView1<f64>) {
    self.reflections = periodic_slopes(params, &BOX);
    self.params = folded(params);
  }

  fn params(&self) -> Array1<f64> {
    self.params.into()
  }

  fn residuals(&mut self) -> Option<Array1<f64>> {
    let model = self.model_prices(&self.params);
    let residuals = self.weighted_residuals_of(&model);
    let options = &self.calibrator.options;
    self.trail.stage(
      options,
      &self.market,
      &model,
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

  fn natural_jacobian(&self) -> Array2<f64> {
    Array2::from_diag(&self.reflections)
  }
}
