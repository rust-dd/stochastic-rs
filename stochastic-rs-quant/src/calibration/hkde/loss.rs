use std::f64::consts::PI;

use ndarray::Array1;
use ndarray::Array2;

use stochastic_rs_distributions::RealExt;

use super::calibrator::HkdeCalibrator;
use super::params::ETA1_MIN;
use super::params::ETA2_MIN;
use super::params::HkdeParams;
use super::params::KAPPA_MIN;
use super::params::P_P_UP;
use super::params::RHO_BOUND;
use super::params::SIGMA_V_MIN;
use super::params::THETA_MIN;
use crate::OptionQuote;
use crate::QuantError;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::calibration::least_squares::Trail;
use crate::pricing::bsm::BsmCoc;
use crate::pricing::bsm::BsmPricer;
use crate::pricing::fourier::HkdeFourier;
use crate::traits::ModelPricer;

const EPS: f64 = 1e-8;

/// One fit: each quote's weight, the current point, the quotes' prices and the trail.
pub(super) struct HkdeProblem<'a> {
  calibrator: &'a HkdeCalibrator,
  sqrt_weights: Vec<f64>,
  params: HkdeParams,
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
      market: calibrator.quotes.iter().map(|quote| quote.price).collect(),
      trail: Trail::negated(),
    })
  }

  pub(super) fn effective_params(&self) -> HkdeParams {
    self.params.projected()
  }

  fn build_model(&self, p: &HkdeParams) -> HkdeFourier {
    HkdeFourier {
      v0: p.v0,
      kappa: p.kappa,
      theta: p.theta,
      sigma_v: p.sigma_v,
      rho: p.rho,
      lam: p.lambda,
      p_up: p.p_up,
      eta1: p.eta1,
      eta2: p.eta2,
    }
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
        price.max_or_nan(0.0)
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
          plus.sigma_v = (x + h).abs().max(SIGMA_V_MIN);
          minus.sigma_v = (x - h).abs().max(SIGMA_V_MIN);
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
    Some(self.numeric_jacobian(&self.effective_params()))
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

/// Reference calibrated Hkde parameters from Agazzotti et al. (2025), Table 1.
///
/// All values correspond to single-name equity option surfaces observed on
/// 2024-02-20. They are exposed as constants so downstream users and tests
/// can check their own implementations against the published results.
///
/// Note: several of these sets do not satisfy the Feller condition and lie
/// outside the default box used by [`HkdeParams::project_in_place`] — the
/// paper does not enforce those constraints. When feeding them through the
/// projection the calibrator will adjust them to the admissible set.
pub mod paper_table1 {
  use super::HkdeParams;

  /// AMZN, 2024-02-20 (Agazzotti et al. 2025, Table 1). Feller violated
  /// ($2\kappa\theta \approx 0.707 < \sigma_v^2 \approx 1.608$) and
  /// $\lambda$, $\sigma_v$ sit outside the default projection box.
  pub const AMZN: HkdeParams = HkdeParams {
    v0: 0.023,
    kappa: 5.275,
    theta: 0.067,
    sigma_v: 1.268,
    rho: -0.691,
    lambda: 53.165,
    p_up: 0.999,
    eta1: 49.799,
    eta2: 2.587,
  };

  /// NFLX, 2024-02-20 (Agazzotti et al. 2025, Table 1). Feller violated,
  /// several parameters exceed the default projection box.
  pub const NFLX: HkdeParams = HkdeParams {
    v0: 0.001,
    kappa: 13.355,
    theta: 0.091,
    sigma_v: 4.797,
    rho: -0.498,
    lambda: 103.622,
    p_up: 0.272,
    eta1: 42.945,
    eta2: 65.011,
  };

  /// SHOP, 2024-02-20 (Agazzotti et al. 2025, Table 1). Fully admissible:
  /// Feller holds ($2\kappa\theta \approx 0.278 \ge \sigma_v^2 \approx 0.038$)
  /// and every parameter lies inside the default projection box.
  pub const SHOP: HkdeParams = HkdeParams {
    v0: 0.176,
    kappa: 0.191,
    theta: 0.728,
    sigma_v: 0.194,
    rho: -0.718,
    lambda: 1.009,
    p_up: 0.958,
    eta1: 8.739,
    eta2: 0.733,
  };

  /// SPOT, 2024-02-20 (Agazzotti et al. 2025, Table 1). Feller violated,
  /// $\lambda$ and $\eta_2$ lie outside the default projection box.
  pub const SPOT: HkdeParams = HkdeParams {
    v0: 0.064,
    kappa: 6.796,
    theta: 0.163,
    sigma_v: 1.698,
    rho: -0.391,
    lambda: 17.725,
    p_up: 1.0,
    eta1: 35.555,
    eta2: 0.049,
  };
}

/// Reference calibration error metrics from Agazzotti et al. (2025), Table 2.
///
/// These values correspond to Hkde calibrated against market option data on
/// 2024-02-20 and are reproduced here as a documentation / sanity aid.
/// Reaching them exactly requires the underlying proprietary quotes.
pub mod paper_table2 {
  /// Hkde mean absolute percentage error per ticker, Table 2.
  pub const MAPE: [(&str, f64); 4] = [
    ("AMZN", 0.0261),
    ("NFLX", 0.0488),
    ("SHOP", 0.0266),
    ("SPOT", 0.0339),
  ];
  /// Hkde root-mean-square error per ticker, Table 2.
  pub const RMSE: [(&str, f64); 4] = [
    ("AMZN", 0.01433),
    ("NFLX", 0.10048),
    ("SHOP", 0.02938),
    ("SPOT", 0.03173),
  ];
}
