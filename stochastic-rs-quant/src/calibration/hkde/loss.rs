use std::f64::consts::PI;

use ndarray::Array1;
use ndarray::Array2;

use super::calibrator::HkdeCalibrator;
use super::params::BOX;
use super::params::ETA1_MIN;
use super::params::ETA2_MIN;
use super::params::KAPPA_MIN;
use super::params::P_P_UP;
use super::params::RHO_BOUND;
use super::params::SIGMA_V_MIN;
use super::params::THETA_MIN;
use crate::OptionQuote;
use crate::QuantError;
use crate::calibration::clip_negative;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::calibration::least_squares::Trail;
use crate::calibration::periodic_slopes;
use crate::calibration::scale_columns;
use crate::pricing::bsm::BsmCoc;
use crate::pricing::bsm::BsmPricer;
use crate::pricing::fourier::HkdeFourier;
use crate::pricing::fourier::HkdeParams;
use crate::traits::ModelPricer;

const EPS: f64 = 1e-8;

pub(super) struct HkdeProblem<'a> {
  calibrator: &'a HkdeCalibrator,
  sqrt_weights: Vec<f64>,
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
      params: start,
      reflections: Array1::ones(9),
      market: calibrator.quotes.iter().map(|quote| quote.price).collect(),
      trail: Trail::negated(),
    })
  }

  pub(super) fn effective_params(&self) -> HkdeParams {
    self.params.projected()
  }

  fn build_model(&self, p: &HkdeParams) -> HkdeFourier {
    HkdeFourier::unchecked(*p)
  }

  pub(super) fn compute_model_prices_for(&self, p: &HkdeParams) -> Vec<f64> {
    let model = self.build_model(p);
    let HkdeCalibrator { s, r, q, .. } = *self.calibrator;
    self
      .calibrator
      .quotes
      .iter()
      .map(|quote| {
        let price = model.price_option(s, quote.strike, r, q, quote.tau, quote.option_type);
        clip_negative(price)
      })
      .collect()
  }

  /// `√w · (model − market)`, the fit's residual; a trail reads its negation.
  fn weighted_residuals_of(&self, model: &[f64]) -> Array1<f64> {
    let rows = self.market.iter().zip(model).zip(&self.sqrt_weights);
    Array1::from_iter(rows.map(|((market, model), w)| w * (model - market)))
  }

  pub(super) fn weighted_residuals_for(&self, p: &HkdeParams) -> Array1<f64> {
    self.weighted_residuals_of(&self.compute_model_prices_for(p))
  }

  /// Central finite-difference Jacobian of the weighted residuals.
  pub(super) fn numeric_jacobian(&self, params: &HkdeParams) -> Array2<f64> {
    let n = self.calibrator.quotes.len();
    let p_dim = 9usize;
    let base_vec: Array1<f64> = (*params).into();
    let mut j_mat = Array2::zeros((n, p_dim));

    for col in 0..p_dim {
      let x = base_vec[col];
      let h = 1e-5_f64.max(1e-3 * x.abs());

      let mut plus = *params;
      let mut minus = *params;
      match col {
        0 => {
          plus.v0 = (x + h).max(0.0);
          minus.v0 = (x - h).max(0.0);
        }
        1 => {
          plus.kappa = (x + h).max(KAPPA_MIN);
          minus.kappa = (x - h).max(KAPPA_MIN);
        }
        2 => {
          plus.theta = (x + h).max(THETA_MIN);
          minus.theta = (x - h).max(THETA_MIN);
        }
        3 => {
          plus.sigma = (x + h).abs().max(SIGMA_V_MIN);
          minus.sigma = (x - h).abs().max(SIGMA_V_MIN);
        }
        4 => {
          plus.rho = (x + h).clamp(-RHO_BOUND, RHO_BOUND);
          minus.rho = (x - h).clamp(-RHO_BOUND, RHO_BOUND);
        }
        5 => {
          plus.lambda = (x + h).max(0.0);
          minus.lambda = (x - h).max(0.0);
        }
        6 => {
          plus.p_up = (x + h).clamp(P_P_UP.0, P_P_UP.1);
          minus.p_up = (x - h).clamp(P_P_UP.0, P_P_UP.1);
        }
        7 => {
          plus.eta1 = (x + h).max(ETA1_MIN);
          minus.eta1 = (x - h).max(ETA1_MIN);
        }
        8 => {
          plus.eta2 = (x + h).max(ETA2_MIN);
          minus.eta2 = (x - h).max(ETA2_MIN);
        }
        _ => unreachable!(),
      }
      plus.project_in_place();
      minus.project_in_place();

      let r_plus = self.weighted_residuals_for(&plus);
      let r_minus = self.weighted_residuals_for(&minus);
      let diff = (r_plus - r_minus) / (2.0 * h);
      for row in 0..n {
        j_mat[(row, col)] = diff[row];
      }
    }

    j_mat
  }
}

impl LeastSquaresProblem for HkdeProblem<'_> {
  fn set_params(&mut self, params: &Array1<f64>) {
    self.reflections = periodic_slopes(params.view(), &BOX);
    self.params = HkdeParams::from(params.clone()).projected();
  }

  fn params(&self) -> Array1<f64> {
    self.effective_params().into()
  }

  fn residuals(&mut self) -> Option<Array1<f64>> {
    let params_eff = self.effective_params();
    let model = self.compute_model_prices_for(&params_eff);
    let residuals = self.weighted_residuals_of(&model);
    let options = &self.calibrator.options;
    self
      .trail
      .stage(options, &self.market, &model, &params_eff, &residuals);
    Some(residuals)
  }

  fn jacobian(&self) -> Option<Array2<f64>> {
    let jacobian = self.numeric_jacobian(&self.effective_params());
    Some(scale_columns(jacobian, self.reflections.view()))
  }

  fn accept(&mut self) {
    self.trail.accept();
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
