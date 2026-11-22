//! Calibration traits — `Calibrator`, `CalibrationResult`, `ToModel`,
//! `ToShortRateModel`.

use super::pricing::ModelPricer;
use crate::CalibrationLossScore;
use crate::calibration::CalibrationHistory;

/// A calibration result that determines a [`ModelPricer`], statically typed, which
/// [`crate::vol_surface::build_surface_from_calibration`] turns into a surface.
pub trait ToModel {
  /// Concrete pricer type produced by this calibration result.
  type Model: ModelPricer;
  /// The pricer a calibration result determines; rates travel with the pricing query, not the
  /// model.
  fn to_model(&self) -> Self::Model;
}

/// Bridge from a short-rate calibration result to its lattice model, built from the result alone:
/// the calibrated parameters and the curve-side rates it was given.
pub trait ToShortRateModel {
  /// Concrete short-rate model produced by this calibration result.
  type Model;

  fn to_short_rate_model(&self) -> Self::Model;
}

/// One entry point per model: `calibrate(start)`; a data-dependent failure is `Err`,
/// non-convergence `Ok(converged = false)`.
pub trait Calibrator {
  /// The parameter set the fit produces and `calibrate` starts from; its result's `Params`.
  type Params: Clone;
  type Output: CalibrationResult<Params = Self::Params>;
  /// Why calibration produced no result; every quant calibrator uses [`crate::QuantError`].
  type Error: std::fmt::Debug + std::fmt::Display + Send + Sync + 'static;

  /// `None` starts from the calibrator's own start; non-convergence is `Ok(converged() == false)`.
  fn calibrate(&self, initial: Option<Self::Params>) -> Result<Self::Output, Self::Error>;
}

/// What a calibration produced; [`Calibrated`](crate::calibration::Calibrated) implements it.
pub trait CalibrationResult {
  type Params: Clone;

  /// Root-mean-square pricing error on the calibration grid.
  fn rmse(&self) -> f64;

  /// Whether the underlying optimiser reported convergence.
  fn converged(&self) -> bool;

  fn params(&self) -> Self::Params;

  /// The loss-metric breakdown; `None` for a calibrator that only tracks `rmse`.
  fn loss_score(&self) -> Option<&CalibrationLossScore> {
    None
  }

  /// Iterations the optimiser ran; `None` when the calibrator does not record them.
  fn iterations(&self) -> Option<usize> {
    None
  }

  /// Why the optimiser stopped; `None` when the calibrator does not surface it.
  fn message(&self) -> Option<&str> {
    None
  }

  /// Worst absolute pricing error on the calibration grid; `None` unless the calibrator records it.
  fn max_error(&self) -> Option<f64> {
    None
  }

  /// The optimiser's trail when history was recorded; empty otherwise.
  fn history(&self) -> &[CalibrationHistory<Self::Params>] {
    &[]
  }
}

#[cfg(test)]
mod tests {
  use super::CalibrationResult;

  struct Bare;

  impl CalibrationResult for Bare {
    type Params = ();

    fn rmse(&self) -> f64 {
      0.1
    }

    fn converged(&self) -> bool {
      true
    }

    fn params(&self) -> Self::Params {}
  }

  #[test]
  fn a_result_without_residuals_has_no_max_error_and_no_trail() {
    assert_eq!(Bare.max_error(), None);
    assert!(Bare.history().is_empty());
  }
}
