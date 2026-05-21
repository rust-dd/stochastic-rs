use super::params::HestonParams;
use crate::CalibrationLossScore;
use crate::LossMetric;

/// Calibration result for the Heston model.
///
/// Wraps the calibrated parameters together with loss / convergence
/// diagnostics so [`HestonCalibrator`](super::calibrator::HestonCalibrator) satisfies the [`Calibrator`](crate::traits::Calibrator)
/// trait.
#[derive(Clone, Debug)]
pub struct HestonCalibrationResult {
  pub params: HestonParams,
  pub loss: CalibrationLossScore,
  pub converged: bool,
}

impl HestonCalibrationResult {
  pub fn to_model(&self, r: f64, q: f64) -> crate::pricing::fourier::HestonFourier {
    self.params.to_model(r, q)
  }
}

impl crate::traits::ToModel for HestonCalibrationResult {
  type Model = crate::pricing::fourier::HestonFourier;
  fn to_model(&self, r: f64, q: f64) -> Self::Model {
    self.params.to_model(r, q)
  }
}

impl crate::traits::CalibrationResult for HestonCalibrationResult {
  type Params = HestonParams;
  fn rmse(&self) -> f64 {
    self.loss.get(LossMetric::Rmse)
  }
  fn converged(&self) -> bool {
    self.converged
  }
  fn params(&self) -> Self::Params {
    self.params.clone()
  }
  fn loss_score(&self) -> Option<&CalibrationLossScore> {
    Some(&self.loss)
  }
}
