//! # Calibration
//!
//! $$
//! \hat\theta=\arg\min_\theta\sum_i w_i\left(P_i^{model}(\theta)-P_i^{mkt}\right)^2
//! $$
//!
use nalgebra::DVector;

use crate::quant::CalibrationLossScore;

pub mod bsm;
pub mod heston;
pub mod rbergomi;
pub mod sabr;
pub mod sabr_smile;

#[derive(Clone, Debug)]
pub struct CalibrationHistory<T> {
  /// Residual vector from calibration objective.
  pub residuals: DVector<f64>,
  pub call_put: DVector<(f64, f64)>,
  /// Model parameter set (input or calibrated output).
  pub params: T,
  /// Calibration loss metric configuration/result.
  pub loss_scores: CalibrationLossScore,
}
