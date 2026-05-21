use super::params::HKDEParams;
use crate::CalibrationLossScore;
use crate::pricing::fourier::HKDEFourier;

/// Calibration result for the Hkde model.
#[derive(Clone, Debug)]
pub struct HKDECalibrationResult {
  pub v0: f64,
  pub kappa: f64,
  pub theta: f64,
  pub sigma_v: f64,
  pub rho: f64,
  pub lambda: f64,
  pub p_up: f64,
  pub eta1: f64,
  pub eta2: f64,
  /// Calibration loss metrics on the unweighted price residuals.
  pub loss: CalibrationLossScore,
  /// Whether the optimiser converged.
  pub converged: bool,
}

impl crate::traits::ToModel for HKDECalibrationResult {
  type Model = HKDEFourier;
  fn to_model(&self, r: f64, q: f64) -> Self::Model {
    HKDECalibrationResult::to_model(self, r, q)
  }
}

impl crate::traits::CalibrationResult for HKDECalibrationResult {
  type Params = HKDEParams;
  fn rmse(&self) -> f64 {
    self.loss.get(crate::LossMetric::Rmse)
  }
  fn converged(&self) -> bool {
    self.converged
  }
  fn params(&self) -> Self::Params {
    HKDEParams {
      v0: self.v0,
      kappa: self.kappa,
      theta: self.theta,
      sigma_v: self.sigma_v,
      rho: self.rho,
      lambda: self.lambda,
      p_up: self.p_up,
      eta1: self.eta1,
      eta2: self.eta2,
    }
  }
  fn loss_score(&self) -> Option<&CalibrationLossScore> {
    Some(&self.loss)
  }
}

impl HKDECalibrationResult {
  /// Convert to a [`HKDEFourier`] model for pricing / vol-surface generation.
  pub fn to_model(&self, r: f64, q: f64) -> HKDEFourier {
    HKDEFourier {
      v0: self.v0,
      kappa: self.kappa,
      theta: self.theta,
      sigma_v: self.sigma_v,
      rho: self.rho,
      r,
      q,
      lam: self.lambda,
      p_up: self.p_up,
      eta1: self.eta1,
      eta2: self.eta2,
    }
  }
}
