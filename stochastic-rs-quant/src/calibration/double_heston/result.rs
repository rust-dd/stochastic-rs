use super::params::DoubleHestonParams;
use crate::CalibrationLossScore;

/// Calibration result for the double Heston model.
#[derive(Clone, Debug)]
pub struct DoubleHestonCalibrationResult {
  pub v1_0: f64,
  pub kappa1: f64,
  pub theta1: f64,
  pub sigma1: f64,
  pub rho1: f64,
  pub v2_0: f64,
  pub kappa2: f64,
  pub theta2: f64,
  pub sigma2: f64,
  pub rho2: f64,
  pub loss: CalibrationLossScore,
  pub converged: bool,
}

impl DoubleHestonCalibrationResult {
  pub fn params(&self) -> DoubleHestonParams {
    DoubleHestonParams {
      v1_0: self.v1_0,
      kappa1: self.kappa1,
      theta1: self.theta1,
      sigma1: self.sigma1,
      rho1: self.rho1,
      v2_0: self.v2_0,
      kappa2: self.kappa2,
      theta2: self.theta2,
      sigma2: self.sigma2,
      rho2: self.rho2,
    }
  }

  pub fn to_model(&self, r: f64, q: f64) -> crate::pricing::fourier::DoubleHestonFourier {
    self.params().to_model(r, q)
  }
}

impl crate::traits::ToModel for DoubleHestonCalibrationResult {
  type Model = crate::pricing::fourier::DoubleHestonFourier;
  fn to_model(&self, r: f64, q: f64) -> Self::Model {
    self.params().to_model(r, q)
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
    DoubleHestonCalibrationResult::params(self)
  }
  fn loss_score(&self) -> Option<&crate::CalibrationLossScore> {
    Some(&self.loss)
  }
}
