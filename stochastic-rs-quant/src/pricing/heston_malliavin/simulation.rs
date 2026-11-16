//! Full-truncation Heston path simulation for Malliavin weights.

use super::HestonMalliavinConfig;
use super::HestonModel;
use super::invalid_input;
use super::variance_vega::effective_initial_variance_bump;
use crate::QuantError;

#[derive(Debug, Clone, Copy)]
pub(super) struct SimulatedPath {
  pub(super) terminal_spot: f64,
  pub(super) integrated_variance: f64,
  pub(super) orthogonal_stochastic_integral: f64,
  pub(super) spot_delta_weight: f64,
  pub(super) spot_gamma_weight: f64,
}

pub(super) fn simulate_path(
  model: HestonModel,
  initial_variance: f64,
  sign: f64,
  variance_normals: &[f64],
  orthogonal_normals: &[f64],
  minimum_integrated_variance: f64,
  minimum_conditional_variance: f64,
) -> Result<SimulatedPath, QuantError> {
  let dt = model.tau / variance_normals.len() as f64;
  let sqrt_dt = dt.sqrt();
  let orthogonal_scale = (1.0 - model.rho.powi(2)).sqrt();
  let mut variance = initial_variance;
  let mut log_spot = model.s.ln();
  let mut integrated_variance = 0.0;
  let mut orthogonal_stochastic_integral = 0.0;

  for (&z_variance, &z_orthogonal) in variance_normals.iter().zip(orthogonal_normals) {
    let z_variance = sign * z_variance;
    let z_orthogonal = sign * z_orthogonal;
    let positive_variance = variance.max(0.0);
    let sqrt_variance = positive_variance.sqrt();
    integrated_variance += positive_variance * dt;
    orthogonal_stochastic_integral += sqrt_variance * sqrt_dt * z_orthogonal;
    log_spot += (model.risk_free_rate - model.dividend_yield - 0.5 * positive_variance) * dt
      + sqrt_variance * sqrt_dt * (model.rho * z_variance + orthogonal_scale * z_orthogonal);
    variance += model.kappa * (model.theta - positive_variance) * dt
      + model.vol_of_vol * sqrt_variance * sqrt_dt * z_variance;
  }

  if integrated_variance <= minimum_integrated_variance {
    return Err(QuantError::Numerical {
      what: "the Malliavin covariance",
      reason: format!(
        "the integrated variance {integrated_variance:e} is at or below its floor \
         {minimum_integrated_variance:e}"
      ),
    });
  }
  let conditional_variance = orthogonal_scale.powi(2) * integrated_variance;
  if conditional_variance <= minimum_conditional_variance {
    return Err(QuantError::Numerical {
      what: "the Malliavin covariance",
      reason: format!(
        "the conditional variance {conditional_variance:e} is at or below its floor \
         {minimum_conditional_variance:e}"
      ),
    });
  }
  let terminal_spot = log_spot.exp();
  if !terminal_spot.is_finite() {
    return Err(QuantError::Numerical {
      what: "the Heston simulation",
      reason: format!("the terminal spot is {terminal_spot}"),
    });
  }
  let conditional_innovation = orthogonal_scale * orthogonal_stochastic_integral;
  let spot_delta_weight = conditional_innovation / (model.s * conditional_variance);
  let spot_gamma_weight = conditional_innovation.powi(2)
    / (model.s.powi(2) * conditional_variance.powi(2))
    - conditional_innovation / (model.s.powi(2) * conditional_variance)
    - 1.0 / (model.s.powi(2) * conditional_variance);
  Ok(SimulatedPath {
    terminal_spot,
    integrated_variance,
    orthogonal_stochastic_integral,
    spot_delta_weight,
    spot_gamma_weight,
  })
}

pub(super) fn validate_model(model: HestonModel) -> Result<(), QuantError> {
  if !model.s.is_finite() || model.s <= 0.0 {
    return Err(invalid_input(format!(
      "spot must be finite and positive, got {}",
      model.s
    )));
  }
  if !model.initial_variance.is_finite() || model.initial_variance <= 0.0 {
    return Err(invalid_input(format!(
      "initial_variance must be finite and positive, got {}",
      model.initial_variance
    )));
  }
  if !model.theta.is_finite() || model.theta < 0.0 {
    return Err(invalid_input(format!(
      "theta must be finite and non-negative, got {}",
      model.theta
    )));
  }
  if !model.kappa.is_finite() || model.kappa < 0.0 {
    return Err(invalid_input(format!(
      "kappa must be finite and non-negative, got {}",
      model.kappa
    )));
  }
  if !model.vol_of_vol.is_finite() || model.vol_of_vol < 0.0 {
    return Err(invalid_input(format!(
      "vol_of_vol must be finite and non-negative, got {}",
      model.vol_of_vol
    )));
  }
  if !model.rho.is_finite() || model.rho.abs() >= 1.0 {
    return Err(invalid_input(format!(
      "the orthogonal Malliavin estimator requires abs(rho) < 1, got rho = {}",
      model.rho
    )));
  }
  if !model.risk_free_rate.is_finite() || !model.dividend_yield.is_finite() {
    return Err(invalid_input(format!(
      "rates must be finite, got r = {}, q = {}",
      model.risk_free_rate, model.dividend_yield
    )));
  }
  if !model.tau.is_finite() || model.tau <= 0.0 {
    return Err(invalid_input(format!(
      "maturity must be finite and positive, got tau = {}",
      model.tau
    )));
  }
  Ok(())
}

pub(super) fn validate_config(
  model: HestonModel,
  config: HestonMalliavinConfig,
) -> Result<(), QuantError> {
  if config.paths < 4 || !config.paths.is_multiple_of(2) {
    return Err(invalid_input(format!(
      "paths must be even and at least four, got {}",
      config.paths
    )));
  }
  if config.steps == 0 {
    return Err(invalid_input("steps must be positive, got 0"));
  }
  effective_initial_variance_bump(
    model.initial_variance,
    config.initial_variance_bump,
    config.minimum_relative_initial_variance_bump,
  )?;
  let floor = config.minimum_integrated_variance;
  if !floor.is_finite() || floor <= 0.0 {
    return Err(invalid_input(format!(
      "minimum_integrated_variance must be finite and positive, got {floor}"
    )));
  }
  let floor = config.minimum_conditional_variance;
  if !floor.is_finite() || floor <= 0.0 {
    return Err(invalid_input(format!(
      "minimum_conditional_variance must be finite and positive, got {floor}"
    )));
  }
  let fraction = config.minimum_orthogonal_variance_fraction;
  if !fraction.is_finite() || fraction <= 0.0 || fraction >= 1.0 {
    return Err(invalid_input(format!(
      "minimum_orthogonal_variance_fraction must lie in (0, 1), got {fraction}"
    )));
  }
  let orthogonal = 1.0 - model.rho.powi(2);
  if orthogonal < fraction {
    return Err(invalid_input(format!(
      "rho leaves too little orthogonal variance for a stable Malliavin score: 1 - rho^2 = \
       {orthogonal:e} is below minimum_orthogonal_variance_fraction = {fraction}"
    )));
  }
  Ok(())
}
