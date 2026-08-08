//! Initial-variance finite-difference controls and reliability diagnostics.

use super::EstimateWithError;
use super::HestonMalliavinError;

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
  pub comparison_estimate: EstimateWithError,
  /// Main estimate minus comparison estimate, with a paired standard error.
  pub bump_difference: EstimateWithError,
  /// Absolute bump difference relative to the larger Greek magnitude.
  pub relative_bump_difference: f64,
  /// Reliability classification used by the strict estimator entry point.
  pub stability: HestonInitialVarianceVegaStability,
}

pub(super) fn effective_initial_variance_bump(
  initial_variance: f64,
  requested_bump: f64,
  minimum_relative_bump: f64,
) -> Result<f64, HestonMalliavinError> {
  if !requested_bump.is_finite() || requested_bump <= 0.0 {
    return Err(HestonMalliavinError::InvalidInput(
      "initial_variance_bump must be finite and positive",
    ));
  }
  if !minimum_relative_bump.is_finite() || !(0.0..1.0).contains(&minimum_relative_bump) {
    return Err(HestonMalliavinError::InvalidInput(
      "minimum_relative_initial_variance_bump must lie in [0, 1)",
    ));
  }
  let effective = requested_bump.max(minimum_relative_bump * initial_variance);
  if effective >= initial_variance {
    return Err(HestonMalliavinError::InvalidInput(
      "effective initial-variance bump must be smaller than initial_variance",
    ));
  }
  Ok(effective)
}

pub(super) fn classify_initial_variance_vega(
  requested_bump: f64,
  minimum_relative_bump: f64,
  effective_bump: f64,
  estimate: EstimateWithError,
  comparison_estimate: EstimateWithError,
  bump_difference: EstimateWithError,
) -> HestonInitialVarianceVegaDiagnostics {
  let scale = estimate
    .value
    .abs()
    .max(comparison_estimate.value.abs())
    .max(f64::MIN_POSITIVE);
  let relative_bump_difference = bump_difference.value.abs() / scale;
  let allowed_difference = STABILITY_STANDARD_ERRORS * bump_difference.standard_error
    + STABILITY_RELATIVE_TOLERANCE * scale;
  let stability = if bump_difference.value.abs() > allowed_difference {
    HestonInitialVarianceVegaStability::BumpSensitive
  } else if estimate.value.abs() <= RESOLUTION_STANDARD_ERRORS * estimate.standard_error {
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
