//! Initial-variance finite-difference controls and reliability diagnostics.

use stochastic_rs_stochastic::mc::McEstimate;

use super::invalid_input;
use crate::QuantError;

pub(super) const DEFAULT_MINIMUM_RELATIVE_BUMP: f64 = 0.03;
const STABILITY_STANDARD_ERRORS: f64 = 3.0;
const STABILITY_RELATIVE_TOLERANCE: f64 = 0.05;
const RESOLUTION_STANDARD_ERRORS: f64 = 2.0;

/// Reliability classification for the CRN initial-variance finite difference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HestonInitialVarianceVegaStability {
  /// The main and half-sized bumps agree and the main estimate resolves zero.
  Stable,
  /// Sampling error is too large to resolve the sign of the variance Greek.
  SamplingUnresolved,
  /// The paired main-minus-half-bump difference is too large to attribute to
  /// sampling error or the documented finite-bump tolerance.
  BumpSensitive,
}

impl HestonInitialVarianceVegaStability {
  /// Returns whether the variance Greek is suitable for a fail-closed caller.
  pub fn is_stable(self) -> bool {
    self == Self::Stable
  }
}

/// Provenance and stability evidence for the CRN initial-variance derivative.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HestonInitialVarianceVegaDiagnostics {
  /// Absolute bump requested in [`super::HestonMalliavinConfig`].
  pub requested_bump: f64,
  /// Relative floor multiplied by the model's initial variance.
  pub minimum_relative_bump: f64,
  /// Bump actually used by the reported variance Greek.
  pub effective_bump: f64,
  /// Half-sized bump evaluated with the same random draws.
  pub comparison_bump: f64,
  /// CRN estimate obtained from the half-sized comparison bump.
  pub comparison_estimate: McEstimate<f64>,
  /// Main estimate minus comparison estimate, with a paired standard error.
  pub bump_difference: McEstimate<f64>,
  /// Absolute bump difference relative to the larger Greek magnitude.
  pub relative_bump_difference: f64,
  /// Reliability classification used by the strict estimator entry point.
  pub stability: HestonInitialVarianceVegaStability,
}

pub(super) fn effective_initial_variance_bump(
  initial_variance: f64,
  requested_bump: f64,
  minimum_relative_bump: f64,
) -> Result<f64, QuantError> {
  if !requested_bump.is_finite() || requested_bump <= 0.0 {
    return Err(invalid_input(format!(
      "initial_variance_bump must be finite and positive, got {requested_bump}"
    )));
  }
  if !minimum_relative_bump.is_finite() || !(0.0..1.0).contains(&minimum_relative_bump) {
    return Err(invalid_input(format!(
      "minimum_relative_initial_variance_bump must lie in [0, 1), got {minimum_relative_bump}"
    )));
  }
  let effective = requested_bump.max(minimum_relative_bump * initial_variance);
  if effective >= initial_variance {
    return Err(invalid_input(format!(
      "effective initial-variance bump must be smaller than initial_variance = \
       {initial_variance}, got {effective}"
    )));
  }
  Ok(effective)
}

pub(super) fn classify_initial_variance_vega(
  requested_bump: f64,
  minimum_relative_bump: f64,
  effective_bump: f64,
  estimate: McEstimate<f64>,
  comparison_estimate: McEstimate<f64>,
  bump_difference: McEstimate<f64>,
) -> HestonInitialVarianceVegaDiagnostics {
  let scale = estimate
    .mean
    .abs()
    .max(comparison_estimate.mean.abs())
    .max(f64::MIN_POSITIVE);
  let relative_bump_difference = bump_difference.mean.abs() / scale;
  let allowed_difference =
    STABILITY_STANDARD_ERRORS * bump_difference.std_err + STABILITY_RELATIVE_TOLERANCE * scale;
  let stability = if bump_difference.mean.abs() > allowed_difference {
    HestonInitialVarianceVegaStability::BumpSensitive
  } else if estimate.mean.abs() <= RESOLUTION_STANDARD_ERRORS * estimate.std_err {
    HestonInitialVarianceVegaStability::SamplingUnresolved
  } else {
    HestonInitialVarianceVegaStability::Stable
  };
  HestonInitialVarianceVegaDiagnostics {
    requested_bump,
    minimum_relative_bump,
    effective_bump,
    comparison_bump: 0.5 * effective_bump,
    comparison_estimate,
    bump_difference,
    relative_bump_difference,
    stability,
  }
}
