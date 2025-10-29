use nalgebra::DVector;

use crate::quant::CalibrationLossScore;

pub mod bsm;
pub mod sabr;
pub mod heston;

#[derive(Clone, Debug)]
pub struct CalibrationHistory<T> {
  pub residuals: DVector<f64>,
  pub call_put: DVector<(f64, f64)>,
  pub params: T,
  pub loss_scores: CalibrationLossScore,
}
