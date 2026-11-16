use super::SvjParams;
use crate::CalibrationLossScore;
use crate::LossMetric;

/// Calibration result for the SVJ (Bates) model.
#[derive(Clone, Debug)]
pub struct SvjCalibrationResult {
  /// Initial variance.
  pub v0: f64,
  /// Mean-reversion speed.
  pub kappa: f64,
  /// Long-run variance.
  pub theta: f64,
  /// Volatility of variance.
  pub sigma_v: f64,
  /// Correlation.
  pub rho: f64,
  /// Jump intensity.
  pub lambda: f64,
  /// Mean log-jump size.
  pub mu_j: f64,
  /// Jump-size volatility.
  pub sigma_j: f64,
  /// Calibration loss metrics.
  pub loss: CalibrationLossScore,
  /// Whether the optimiser converged.
  pub converged: bool,
}

impl crate::traits::ToModel for SvjCalibrationResult {
  type Model = crate::pricing::fourier::BatesFourier;
  fn to_model(&self) -> Self::Model {
    crate::pricing::fourier::BatesFourier {
      v0: self.v0,
      kappa: self.kappa,
      theta: self.theta,
      sigma_v: self.sigma_v,
      rho: self.rho,
      lambda: self.lambda,
      mu_j: self.mu_j,
      sigma_j: self.sigma_j,
    }
  }
}

impl crate::traits::CalibrationResult for SvjCalibrationResult {
  type Params = SvjParams;
  fn rmse(&self) -> f64 {
    self.loss.get(LossMetric::Rmse)
  }

  fn converged(&self) -> bool {
    self.converged
  }

  fn params(&self) -> Self::Params {
    SvjParams {
      v0: self.v0,
      kappa: self.kappa,
      theta: self.theta,
      sigma_v: self.sigma_v,
      rho: self.rho,
      lambda: self.lambda,
      mu_j: self.mu_j,
      sigma_j: self.sigma_j,
    }
  }

  fn loss_score(&self) -> Option<&CalibrationLossScore> {
    Some(&self.loss)
  }
}
