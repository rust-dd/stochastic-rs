use std::f64::consts::PI;

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

use super::calibrator::HkdeCalibrator;
use super::params::BOX;
use super::params::BOXED;
use crate::OptionQuote;
use crate::OptionType;
use crate::QuantError;
use crate::calibration::clip_negative;
use crate::calibration::feller_share::natural_jacobian;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::calibration::least_squares::Trail;
use crate::calibration::periodic_slopes;
use crate::calibration::scale_columns;
use crate::calibration::stencil::derivative;
use crate::calibration::stencil::step;
use crate::pricing::bsm::BsmCoc;
use crate::pricing::bsm::BsmPricer;
use crate::pricing::cf_quadrature::Rule;
use crate::pricing::cf_quadrature::Walk;
use crate::pricing::fourier::HkdeFourier;
use crate::pricing::fourier::HkdeParams;
use crate::pricing::fourier::LewisPricer;

const EPS: f64 = 1e-8;

pub(super) struct HkdeProblem<'a> {
  calibrator: &'a HkdeCalibrator,
  sqrt_weights: Vec<f64>,
  /// The folded point: `sigma` holds σ's share of the way to Feller's cap.
  params: HkdeParams,
  /// `dθⱼ/dxⱼ` of the box's reflection at the optimiser's raw coordinates.
  reflections: Array1<f64>,
  pub(super) market: Vec<f64>,
  pub(super) trail: Trail<HkdeParams>,
}

impl<'a> HkdeProblem<'a> {
  /// `InvalidQuote` for the first quote `market_weight` cannot weight.
  pub(super) fn new(calibrator: &'a HkdeCalibrator, start: HkdeParams) -> Result<Self, QuantError> {
    let HkdeCalibrator { s, r, q, .. } = *calibrator;
    let sqrt_weights = calibrator
      .quotes
      .iter()
      .enumerate()
      .map(|(index, quote)| {
        market_weight(s, r, q, quote)
          .map(f64::sqrt)
          .map_err(|reason| QuantError::InvalidQuote { index, reason })
      })
      .collect::<Result<Vec<_>, _>>()?;
    Ok(Self {
      calibrator,
      sqrt_weights,
      params: start.to_share(),
      reflections: Array1::ones(9),
      market: calibrator.quotes.iter().map(|quote| quote.price).collect(),
      trail: Trail::negated(),
    })
  }

  pub(super) fn effective_params(&self) -> HkdeParams {
    self.params.at_share()
  }

  fn build_model(&self, p: &HkdeParams) -> HkdeFourier {
    HkdeFourier::unchecked(*p)
  }

  pub(super) fn compute_model_prices_for(&self, p: &HkdeParams) -> Vec<f64> {
    let quotes = self.calibrator.quotes.iter();
    quotes
      .map(|quote| self.price(p, quote, &mut Walk::Adaptive))
      .collect()
  }

  /// The quote's Lewis price, a put by parity, as `ModelPricer::price_option` gives it.
  fn price(&self, p: &HkdeParams, quote: &OptionQuote, walk: &mut Walk<'_, Rule>) -> f64 {
    let HkdeCalibrator { s, r, q, .. } = *self.calibrator;
    let (k, tau) = (quote.strike, quote.tau);
    let call = LewisPricer::price_call_walk(&self.build_model(p), s, k, r, q, tau, walk);
    clip_negative(match quote.option_type {
      OptionType::Call => call,
      OptionType::Put => call - s * (-q * tau).exp() + k * (-r * tau).exp(),
    })
  }

  /// `√w · (model − market)`, the fit's residual; a trail reads its negation.
  pub(super) fn weighted_residuals_of(&self, model: &[f64]) -> Array1<f64> {
    let rows = self.market.iter().zip(model).zip(&self.sqrt_weights);
    Array1::from_iter(rows.map(|((market, model), w)| w * (model - market)))
  }

  /// `∂r/∂x` at the folded point, each stencil point through σ's share and every price on the
  /// rules the point's own prices walked; `scale` times the shared step.
  pub(super) fn difference_jacobian(&self, scale: f64) -> Array2<f64> {
    let point = Array1::from(self.params);
    let quotes = &self.calibrator.quotes;
    let at = self.effective_params();
    let rules = quotes.iter().map(|quote| {
      let mut rules = Vec::new();
      self.price(&at, quote, &mut Walk::Record(&mut rules));
      rules
    });
    let rules = rules.collect::<Vec<_>>();
    let mut jacobian = Array2::zeros((quotes.len(), point.len()));
    for (j, mut column) in jacobian.columns_mut().into_iter().enumerate() {
      let x = point[j];
      column.assign(&derivative(x, scale * step(x), BOXED[j], |value| {
        let mut moved = point.clone();
        moved[j] = value;
        let moved = HkdeParams::from(moved).at_share();
        let replay = quotes.iter().zip(&rules);
        let model =
          replay.map(|(quote, rules)| self.price(&moved, quote, &mut Walk::Replay(rules.iter())));
        self.weighted_residuals_of(&model.collect::<Vec<_>>())
      }));
    }
    jacobian
  }
}

impl LeastSquaresProblem for HkdeProblem<'_> {
  fn set_params(&mut self, params: ArrayView1<f64>) {
    self.reflections = periodic_slopes(params, &BOX);
    self.params = HkdeParams::from(params.to_owned()).folded();
  }

  fn params(&self) -> Array1<f64> {
    self.params.into()
  }

  fn residuals(&mut self) -> Option<Array1<f64>> {
    let params_eff = self.effective_params();
    let model = self.compute_model_prices_for(&params_eff);
    let residuals = self.weighted_residuals_of(&model);
    let options = &self.calibrator.options;
    self
      .trail
      .stage(options, &self.market, &model, &params_eff, residuals.view());
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
    Array1::from(self.effective_params()).to_vec()
  }

  fn natural_jacobian(&self) -> Array2<f64> {
    natural_jacobian(self.reflections.view(), &[(1, self.params.share_slopes())])
  }
}

/// Eq. 13's weight $w_j^{(n)} = (S_0\,\psi(d_1)\,\sqrt{T_n})^{-1}$ at the quote's implied
/// volatility at the carry `r − q`; the reason, as `Err`, when the weights cannot use the quote.
pub(super) fn market_weight(s: f64, r: f64, q: f64, quote: &OptionQuote) -> Result<f64, String> {
  // The pricer's own volatility is unused: only its carry convention enters the inversion.
  let sigma = BsmPricer::new(0.2, BsmCoc::Merton1973).implied_volatility(
    quote.price,
    s,
    quote.strike,
    r,
    q,
    quote.tau,
    quote.option_type,
  );
  weight_at(s, r, q, quote, sigma)
}

/// Eq. 13's weight at the implied volatility `sigma`; `Err` unless it and the vega exceed 1e-8.
pub(super) fn weight_at(
  s: f64,
  r: f64,
  q: f64,
  quote: &OptionQuote,
  sigma: f64,
) -> Result<f64, String> {
  if !(sigma.is_finite() && sigma > EPS) {
    return Err(format!(
      "no Black volatility above 1e-8 reprices its price {:?} at the carry r − q = {:?}",
      quote.price,
      r - q
    ));
  }
  let (k, t) = (quote.strike, quote.tau);
  let sqrt_t = t.sqrt();
  let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
  let phi = (-0.5 * d1 * d1).exp() / (2.0 * PI).sqrt();
  let vega = s * phi * sqrt_t;
  if !(vega.is_finite() && vega >= EPS) {
    return Err(format!(
      "its vega S·ψ(d1)·√T = {vega:?} at the implied volatility {sigma:?} is below 1e-8"
    ));
  }
  Ok(1.0 / vega)
}
