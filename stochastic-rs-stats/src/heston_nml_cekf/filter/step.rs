//! One-step CEKF covariance and state algebra.

use super::HestonCekfCorrection;
use super::HestonCekfFilterConfig;
use super::HestonCekfPositiveStatePolicy;
use super::HestonCekfState;
use super::HestonCekfStepDiagnostics;
use super::HestonCekfStepResult;
use super::HestonCekfVarianceProjection;
use crate::heston_nml_cekf::EPS;
use crate::heston_nml_cekf::HestonCekfError;
use crate::heston_nml_cekf::HestonNMLECEKFParams;

#[derive(Clone, Copy)]
pub(super) enum FloorPolicy {
  Error,
  Clamp,
}

pub(super) fn step_impl(
  previous: HestonCekfState,
  log_return: f64,
  parameters: HestonNMLECEKFParams,
  config: &HestonCekfFilterConfig,
  floor_policy: FloorPolicy,
) -> Result<HestonCekfStepResult, HestonCekfError> {
  let variance = previous.variance.max(EPS);
  let covariance = previous.error_covariance_bound.max(EPS);
  let state_transition = 1.0 - parameters.kappa * config.delta;
  let diffusion_loading = parameters.sigma * (variance * config.delta).max(EPS).sqrt();
  let process_variance = config.q22 * diffusion_loading * diffusion_loading;

  let delta_q = match config.correction {
    HestonCekfCorrection::Traditional => 0.0,
    HestonCekfCorrection::Consistent { bounds } => {
      let upper_bound = covariance * bounds.max_abs_state_transition.powi(2)
        + (config.delta * bounds.max_abs_kappa_theta).powi(2)
        + bounds.max_abs_sigma.powi(2) * config.delta * variance * config.q22;
      (upper_bound - (state_transition.powi(2) * covariance + process_variance)).max(0.0)
    }
  };
  let predicted_covariance = apply_floor(
    state_transition.powi(2) * covariance + process_variance + delta_q,
    "predicted error covariance bound",
    floor_policy,
  )?;
  let predicted_variance = apply_floor(
    variance + parameters.kappa * (parameters.theta - variance) * config.delta,
    "predicted variance",
    floor_policy,
  )?;

  let measurement_jacobian = -0.5 * config.delta;
  let sqrt_variance_delta = (predicted_variance * config.delta).max(EPS).sqrt();
  let orthogonal_rho = (1.0 - parameters.rho * parameters.rho).max(0.0).sqrt();
  let measurement_loading_1 = orthogonal_rho * sqrt_variance_delta;
  let measurement_loading_2 = parameters.rho * sqrt_variance_delta;
  let measurement_variance = config.q11 * measurement_loading_1.powi(2)
    + 2.0 * config.q12 * measurement_loading_1 * measurement_loading_2
    + config.q22 * measurement_loading_2.powi(2);
  let process_measurement_cross_covariance =
    diffusion_loading * (config.q12 * measurement_loading_1 + config.q22 * measurement_loading_2);
  let innovation_covariance = apply_floor(
    measurement_jacobian.powi(2) * predicted_covariance
      + measurement_variance
      + 2.0 * measurement_jacobian * process_measurement_cross_covariance,
    "innovation covariance",
    floor_policy,
  )?;
  let covariance_innovation_cross =
    measurement_jacobian * predicted_covariance + process_measurement_cross_covariance;
  let kalman_gain = covariance_innovation_cross / innovation_covariance;
  let innovation = log_return - (config.r - 0.5 * predicted_variance) * config.delta;
  let raw_updated_variance = predicted_variance + kalman_gain * innovation;
  let (updated_variance, updated_variance_projection) =
    constrain_updated_variance(raw_updated_variance, config, floor_policy)?;

  let delta_r = match config.correction {
    HestonCekfCorrection::Traditional => 0.0,
    HestonCekfCorrection::Consistent { .. } => {
      let q_mix = orthogonal_rho.powi(2) * config.q11
        + 2.0 * parameters.rho * orthogonal_rho * config.q12
        + parameters.rho.powi(2) * config.q22;
      let upper_bound = predicted_covariance * (1.0 + 0.5 * kalman_gain * config.delta).powi(2)
        + 2.0 * kalman_gain.powi(2) * config.delta * predicted_variance * q_mix
        - predicted_covariance
        + kalman_gain * covariance_innovation_cross;
      upper_bound.max(0.0)
    }
  };
  let updated_covariance = apply_floor(
    predicted_covariance - kalman_gain * covariance_innovation_cross + delta_r,
    "updated error covariance bound",
    floor_policy,
  )?;

  Ok(HestonCekfStepResult {
    state: HestonCekfState {
      variance: updated_variance,
      error_covariance_bound: updated_covariance,
    },
    diagnostics: HestonCekfStepDiagnostics {
      predicted_variance,
      predicted_error_covariance_bound: predicted_covariance,
      innovation,
      innovation_covariance,
      kalman_gain,
      process_measurement_cross_covariance,
      delta_q,
      delta_r,
      updated_variance_projection,
    },
  })
}

fn constrain_updated_variance(
  raw_variance: f64,
  config: &HestonCekfFilterConfig,
  floor_policy: FloorPolicy,
) -> Result<(f64, Option<HestonCekfVarianceProjection>), HestonCekfError> {
  if matches!(floor_policy, FloorPolicy::Clamp) {
    return Ok((raw_variance.max(EPS), None));
  }
  match config.positive_state_policy {
    HestonCekfPositiveStatePolicy::Strict => Ok((
      apply_floor(raw_variance, "updated variance", floor_policy)?,
      None,
    )),
    HestonCekfPositiveStatePolicy::Project { floor } => {
      if !raw_variance.is_finite() {
        return Err(HestonCekfError::FilterFloorHit {
          quantity: "updated variance",
          value: raw_variance,
        });
      }
      if raw_variance >= floor {
        return Ok((raw_variance, None));
      }
      Ok((
        floor,
        Some(HestonCekfVarianceProjection {
          raw_variance,
          projected_variance: floor,
        }),
      ))
    }
  }
}

fn apply_floor(
  value: f64,
  quantity: &'static str,
  floor_policy: FloorPolicy,
) -> Result<f64, HestonCekfError> {
  if value.is_finite() && value > EPS {
    return Ok(value);
  }
  match floor_policy {
    FloorPolicy::Error => Err(HestonCekfError::FilterFloorHit { quantity, value }),
    FloorPolicy::Clamp => Ok(value.max(EPS)),
  }
}
