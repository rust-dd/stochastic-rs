//! Full-truncation Heston path simulation for Malliavin weights.

use stochastic_rs_core::simd_rng::SeedExt;

use super::HestonMalliavinConfig;
use super::HestonModel;
use super::variance_vega::effective_initial_variance_bump;
use crate::QuantError;
use crate::pricing::heston::HestonParams;

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
  let orthogonal_scale = (1.0 - model.params.rho.powi(2)).sqrt();
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
      + sqrt_variance * sqrt_dt * (model.params.rho * z_variance + orthogonal_scale * z_orthogonal);
    variance += model.params.kappa * (model.params.theta - positive_variance) * dt
      + model.params.sigma * sqrt_variance * sqrt_dt * z_variance;
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

pub(super) fn assert_model(model: HestonModel) {
  let HestonModel {
    s,
    params,
    risk_free_rate,
    dividend_yield,
    tau,
  } = model;
  assert!(
    s.is_finite() && s > 0.0,
    "s must satisfy `s.is_finite() && s > 0.0`, got s = {s:?}"
  );
  assert!(
    params.v0 > 0.0,
    "v0 must satisfy `v0 > 0.0`, got v0 = {:?}",
    params.v0
  );
  let HestonParams { kappa, rho, .. } = params.checked();
  assert!(
    kappa >= 0.0,
    "kappa must satisfy `kappa >= 0.0`, got kappa = {kappa:?}"
  );
  assert!(
    rho.abs() < 1.0,
    "rho must satisfy `rho.abs() < 1.0`, got rho = {rho:?}"
  );
  assert!(
    risk_free_rate.is_finite(),
    "risk_free_rate must satisfy `risk_free_rate.is_finite()`, got risk_free_rate = \
     {risk_free_rate:?}"
  );
  assert!(
    dividend_yield.is_finite(),
    "dividend_yield must satisfy `dividend_yield.is_finite()`, got dividend_yield = \
     {dividend_yield:?}"
  );
  assert!(
    tau.is_finite() && tau > 0.0,
    "tau must satisfy `tau.is_finite() && tau > 0.0`, got tau = {tau:?}"
  );
}

pub(super) fn assert_config<S: SeedExt>(model: HestonModel, config: &HestonMalliavinConfig<S>) {
  let HestonMalliavinConfig {
    paths,
    steps,
    initial_variance_bump: bump,
    minimum_relative_initial_variance_bump: relative,
    minimum_integrated_variance: integrated,
    minimum_conditional_variance: conditional,
    minimum_orthogonal_variance_fraction: fraction,
    ..
  } = *config;
  assert!(
    paths >= 4 && paths.is_multiple_of(2),
    "paths must satisfy `paths >= 4 && paths.is_multiple_of(2)`, got paths = {paths:?}"
  );
  assert!(
    steps >= 1,
    "steps must satisfy `steps >= 1`, got steps = {steps:?}"
  );
  assert!(
    bump.is_finite() && bump > 0.0,
    "initial_variance_bump must satisfy `initial_variance_bump.is_finite() && \
     initial_variance_bump > 0.0`, got initial_variance_bump = {bump:?}"
  );
  assert!(
    (0.0..1.0).contains(&relative),
    "minimum_relative_initial_variance_bump must satisfy \
     `(0.0..1.0).contains(&minimum_relative_initial_variance_bump)`, got \
     minimum_relative_initial_variance_bump = {relative:?}"
  );
  let effective = effective_initial_variance_bump(model.params.v0, bump, relative);
  assert!(
    effective < model.params.v0,
    "the effective initial-variance bump must satisfy `max(initial_variance_bump, \
     minimum_relative_initial_variance_bump * v0) < v0`, got {effective:?} against v0 = {:?}",
    model.params.v0
  );
  assert!(
    integrated.is_finite() && integrated > 0.0,
    "minimum_integrated_variance must satisfy `minimum_integrated_variance.is_finite() && \
     minimum_integrated_variance > 0.0`, got minimum_integrated_variance = {integrated:?}"
  );
  assert!(
    conditional.is_finite() && conditional > 0.0,
    "minimum_conditional_variance must satisfy `minimum_conditional_variance.is_finite() && \
     minimum_conditional_variance > 0.0`, got minimum_conditional_variance = {conditional:?}"
  );
  assert!(
    fraction > 0.0 && fraction < 1.0,
    "minimum_orthogonal_variance_fraction must satisfy `minimum_orthogonal_variance_fraction > \
     0.0 && minimum_orthogonal_variance_fraction < 1.0`, got \
     minimum_orthogonal_variance_fraction = {fraction:?}"
  );
  let rho = model.params.rho;
  assert!(
    1.0 - rho.powi(2) >= fraction,
    "rho must satisfy `1.0 - rho.powi(2) >= minimum_orthogonal_variance_fraction`, got rho = \
     {rho:?} against minimum_orthogonal_variance_fraction = {fraction:?}"
  );
}
