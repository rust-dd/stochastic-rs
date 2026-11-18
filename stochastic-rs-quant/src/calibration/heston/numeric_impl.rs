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
use super::transform::from_optimizer_coordinates;
use super::transform::to_optimizer_coordinates;
use crate::OptionType;
use crate::calibration::clip_negative;
use crate::pricing::heston::HestonParams;
use crate::pricing::heston::HestonPricer;

/// Overshoot weight per unit of the price residuals' σ sensitivity: boundary fits keep their optimum from 0.01 to 0.3
/// and lose it from 1 up, where the row stiffens the kink at the cap.
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
      let (call, put) = pricer.call_put(self.s, quote.strike, self.r, self.q, quote.tau);

      match quote.option_type {
        OptionType::Call => c_model[idx] = clip_negative(call),
        OptionType::Put => c_model[idx] = clip_negative(put),
      }
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

  /// Numerically approximate the residual Jacobian with respect to physical parameters.
  #[cfg(test)]
  #[allow(non_snake_case)]
  pub(super) fn numeric_jacobian(&self, params: &HestonParams) -> Array2<f64> {
    self.finite_difference_jacobian(params, false)
  }

  pub(super) fn numeric_optimizer_jacobian(&self, params: &HestonParams) -> Array2<f64> {
    self.finite_difference_jacobian(params, true)
  }

  #[allow(non_snake_case)]
  fn finite_difference_jacobian(
    &self,
    params: &HestonParams,
    optimizer_denominator: bool,
  ) -> Array2<f64> {
    let n = self.quotes.len();
    let p = 5usize;
    let coordinates = to_optimizer_coordinates(params);
    let mut J = Array2::zeros((n, p));
    // A physical column divides by its own parameter's change, so it differentiates the unprojected prices.
    let project = |params: HestonParams| {
      if optimizer_denominator {
        self.feller_projected(params)
      } else {
        params
      }
    };

    for col in 0..p {
      let h = 1e-5 * (1.0 + coordinates[col].abs());
      let mut plus = coordinates.clone();
      let mut minus = coordinates.clone();
      plus[col] += h;
      minus[col] -= h;
      let params_plus = project(from_optimizer_coordinates(plus.view()));
      let params_minus = project(from_optimizer_coordinates(minus.view()));

      let r_plus = self.residuals_for(&params_plus);
      let r_minus = self.residuals_for(&params_minus);
      let denominator = if optimizer_denominator {
        2.0 * h
      } else {
        let plus_physical = Array1::from(params_plus);
        let minus_physical = Array1::from(params_minus);
        plus_physical[col] - minus_physical[col]
      };
      let diff = (r_plus - r_minus) / denominator;
      for row in 0..n {
        J[(row, col)] = diff[row];
      }
    }

    J
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
