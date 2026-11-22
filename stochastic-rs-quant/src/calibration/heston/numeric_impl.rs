use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use stochastic_rs_stats::StatsError;
use stochastic_rs_stats::heston_mle::fit_heston_nmle;
use stochastic_rs_stats::heston_mle::fit_heston_nmle_with_delta;
use stochastic_rs_stats::heston_mle::fit_heston_pmle;
use stochastic_rs_stats::heston_mle::fit_heston_pmle_with_delta;
use stochastic_rs_stats::heston_nmle_cekf::fit_heston_nmle_cekf;

use super::calibrator::HestonCalibrator;
use super::calibrator::HestonMleSeed;
use super::params::HestonJacobianMethod;
use super::params::HestonMleSeedMethod;
use super::transform::BOUNDS;
use super::transform::apply_chain_rule;
use super::transform::to_optimizer_coordinates;
use crate::calibration::clip_negative;
use crate::calibration::stencil::derivative;
use crate::calibration::stencil::step;
use crate::pricing::heston::HestonParams;
use crate::pricing::heston::HestonPricer;
use crate::traits::ModelPricer;

/// Overshoot weight per unit of the price residuals' σ sensitivity: boundary fits keep their
/// optimum from 0.01 to 0.3 and lose it from 1 up, where the row stiffens the kink at the cap.
const FELLER_ROW_SCALE: f64 = 0.1;

impl HestonCalibrator {
  pub(super) fn fallback_params() -> HestonParams {
    HestonParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.5,
      rho: -0.5,
    }
    .projected()
  }

  /// The overshoot row's weight, scaled to the norm of the price rows' σ column at `start`; the
  /// difference runs backwards, since lowering σ keeps the point feasible.
  pub(super) fn feller_row_weight_at(&self, start: HestonParams) -> f64 {
    let step = 1e-5 * (1.0 + start.sigma);
    let lowered = HestonParams {
      sigma: start.sigma - step,
      ..start
    };
    let column = (self.residuals_for(&start) - self.residuals_for(&lowered)) / step;
    let norm = Some(column.dot(&column).sqrt()).filter(|norm| norm.is_finite() && *norm > 0.0);
    FELLER_ROW_SCALE * norm.unwrap_or(1.0)
  }

  pub(super) fn feller_projected(&self, params: HestonParams) -> HestonParams {
    if self.feller_projection {
      params.projected_with_feller_condition()
    } else {
      params
    }
  }

  pub(super) fn compute_model_prices_for_numeric(&self, params: &HestonParams) -> Array1<f64> {
    let mut c_model = Array1::zeros(self.quotes.len());

    for (idx, quote) in self.quotes.iter().enumerate() {
      let pricer = HestonPricer::new(*params);
      let price = pricer.price_option(
        self.s,
        quote.strike,
        self.r,
        self.q,
        quote.tau,
        quote.option_type,
      );
      c_model[idx] = clip_negative(price);
    }

    c_model
  }

  pub(super) fn compute_model_prices_for(&self, params: &HestonParams) -> Array1<f64> {
    match self.jacobian_method {
      HestonJacobianMethod::NumericFiniteDiff => self.compute_model_prices_for_numeric(params),
      HestonJacobianMethod::CuiAnalytic => self
        .compute_model_prices_for_cui(params)
        .unwrap_or_else(|| self.compute_model_prices_for_numeric(params)),
    }
  }

  pub(super) fn residuals_for(&self, params: &HestonParams) -> Array1<f64> {
    self.weighted_residuals(self.compute_model_prices_for(params).view())
  }

  pub(super) fn weighted_residuals(&self, model: ArrayView1<f64>) -> Array1<f64> {
    Array1::from_iter(
      self
        .quotes
        .iter()
        .zip(model)
        .enumerate()
        .map(|(idx, (quote, model))| (quote.price - model) * self.weight(idx)),
    )
  }

  pub(super) fn weight(&self, idx: usize) -> f64 {
    self.residual_weights.as_ref().map_or(1.0, |w| w[idx])
  }

  /// The residual Jacobian in the physical parameters, at the unprojected prices.
  #[cfg(test)]
  pub(super) fn numeric_jacobian(&self, params: &HestonParams) -> Array2<f64> {
    self.physical_jacobian(params, false, 1.0)
  }

  /// In the optimiser's coordinates, through Feller's projection from the free point.
  pub(super) fn numeric_optimizer_jacobian(&self, params: &HestonParams) -> Array2<f64> {
    let physical = self.physical_jacobian(params, true, 1.0);
    apply_chain_rule(physical, to_optimizer_coordinates(params).view())
  }

  /// `∂r/∂θ` at `params`, each stencil inside the box and projected where `projected` says;
  /// `scale` times the shared step.
  pub(super) fn physical_jacobian(
    &self,
    params: &HestonParams,
    projected: bool,
    scale: f64,
  ) -> Array2<f64> {
    let point = [
      params.v0,
      params.kappa,
      params.theta,
      params.sigma,
      params.rho,
    ];
    let mut jacobian = Array2::zeros((self.quotes.len(), point.len()));
    for (j, mut column) in jacobian.columns_mut().into_iter().enumerate() {
      let x = point[j];
      column.assign(&derivative(x, scale * step(x), BOUNDS[j], |value| {
        let mut moved = point;
        moved[j] = value;
        let [v0, kappa, theta, sigma, rho] = moved;
        let moved = HestonParams {
          v0,
          kappa,
          theta,
          sigma,
          rho,
        };
        let moved = if projected {
          self.feller_projected(moved)
        } else {
          moved
        };
        self.residuals_for(&moved)
      }));
    }
    jacobian
  }
}

impl HestonMleSeed {
  /// The estimator's fit to the series, at `r` or the calibrator's `rate` when it is `None`.
  pub(super) fn fit(&self, rate: f64) -> Result<HestonParams, StatsError> {
    let (s, v, r) = (self.s.view(), self.v.view(), self.r.unwrap_or(rate));
    let fit = match (self.method, self.delta) {
      (HestonMleSeedMethod::Nmle, None) => fit_heston_nmle(s, v, r)?,
      (HestonMleSeedMethod::Nmle, Some(delta)) => fit_heston_nmle_with_delta(s, v, r, delta)?,
      (HestonMleSeedMethod::Pmle, None) => fit_heston_pmle(s, v, r)?,
      (HestonMleSeedMethod::Pmle, Some(delta)) => fit_heston_pmle_with_delta(s, v, r, delta)?,
      (HestonMleSeedMethod::NmleCekf, delta) => {
        let mut config = self.nmle_cekf.clone().unwrap_or_default();
        config.r = r;
        config.delta = delta.unwrap_or(1.0 / self.s.len().saturating_sub(1).max(1) as f64);
        if let Some(&v0) = self.v.first()
          && v0.is_finite()
          && v0 > 0.0
        {
          config.initial_v0 = v0;
        }
        fit_heston_nmle_cekf(s, config)?.params
      }
    };
    Ok(fit.into())
  }
}
