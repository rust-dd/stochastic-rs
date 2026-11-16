use super::params::DoubleHestonParams;
use crate::CalibrationLossScore;
use crate::traits::ToModel;

/// Calibration result for the double Heston model.
#[derive(Clone, Debug)]
pub struct DoubleHestonCalibrationResult {
  pub params: DoubleHestonParams,
  pub loss: CalibrationLossScore,
  pub converged: bool,
}

impl ToModel for DoubleHestonCalibrationResult {
  type Model = crate::pricing::fourier::DoubleHestonFourier;
  fn to_model(&self) -> Self::Model {
    self.params.to_model()
  }
}

impl crate::traits::CalibrationResult for DoubleHestonCalibrationResult {
  type Params = DoubleHestonParams;
  fn rmse(&self) -> f64 {
    self.loss.get(crate::LossMetric::Rmse)
  }

  fn converged(&self) -> bool {
    self.converged
  }

  fn params(&self) -> Self::Params {
    self.params
  }

  fn loss_score(&self) -> Option<&crate::CalibrationLossScore> {
    Some(&self.loss)
  }
}
