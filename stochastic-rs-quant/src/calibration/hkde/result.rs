use super::params::HkdeParams;
use crate::CalibrationLossScore;
use crate::pricing::fourier::HkdeFourier;

/// Calibration result for the Hkde model.
#[derive(Clone, Debug)]
pub struct HkdeCalibrationResult {
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

impl crate::traits::ToModel for HkdeCalibrationResult {
  type Model = HkdeFourier;
  fn to_model(&self, r: f64, q: f64) -> Self::Model {
    HkdeCalibrationResult::to_model(self, r, q)
  }
}

impl crate::traits::CalibrationResult for HkdeCalibrationResult {
  type Params = HkdeParams;
  fn rmse(&self) -> f64 {
    self.loss.get(crate::LossMetric::Rmse)
  }

  fn converged(&self) -> bool {
    self.converged
  }

  fn params(&self) -> Self::Params {
    HkdeParams {
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

impl HkdeCalibrationResult {
  /// Convert to a [`HkdeFourier`] model for pricing / vol-surface generation.
  pub fn to_model(&self, r: f64, q: f64) -> HkdeFourier {
    HkdeFourier {
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
