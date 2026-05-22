use nalgebra::DVector;

use super::calibrator::KAPPA_MIN;
use super::calibrator::SIGMA_V_MIN;
use super::calibrator::THETA_MIN;

pub(super) const EPS: f64 = 1e-8;
pub(super) const RHO_BOUND: f64 = 0.9999;

pub(super) const P_V0: (f64, f64) = (0.005, 0.25);
pub(super) const P_KAPPA: (f64, f64) = (0.1, 20.0);
pub(super) const P_THETA: (f64, f64) = (0.001, 0.4);
pub(super) const P_SIGMA_V: (f64, f64) = (0.01, 1.0);
pub(super) const P_RHO: (f64, f64) = (-1.0, 1.0);
pub(super) const P_LAMBDA: (f64, f64) = (0.0, 10.0);
pub(super) const P_MU_J: (f64, f64) = (-0.5, 0.5);
pub(super) const P_SIGMA_J: (f64, f64) = (0.001, 1.0);

/// SVJ / Bates model parameters.
#[derive(Clone, Copy, Debug)]
pub struct SVJParams {
  /// Initial variance $v_0$.
  pub v0: f64,
  /// Mean-reversion speed $\kappa$.
  pub kappa: f64,
  /// Long-run variance $\theta$.
  pub theta: f64,
  /// Volatility of variance $\sigma_v$.
  pub sigma_v: f64,
  /// Correlation $\rho$ between price and variance Brownian motions.
  pub rho: f64,
  /// Jump intensity $\lambda$ (Poisson arrival rate).
  pub lambda: f64,
  /// Mean log-jump size $\mu_J$.
  pub mu_j: f64,
  /// Jump-size volatility $\sigma_J$.
  pub sigma_j: f64,
}

impl SVJParams {
  /// Project parameters to satisfy admissibility constraints (box + Feller condition).
  pub fn project_in_place(&mut self) {
    use crate::calibration::periodic_map;

    self.v0 = periodic_map(self.v0, P_V0.0, P_V0.1).max(0.0);
    self.kappa = periodic_map(self.kappa, P_KAPPA.0, P_KAPPA.1).max(KAPPA_MIN);
    self.theta = periodic_map(self.theta, P_THETA.0, P_THETA.1).max(THETA_MIN);
    self.sigma_v = periodic_map(self.sigma_v, P_SIGMA_V.0, P_SIGMA_V.1)
      .abs()
      .max(SIGMA_V_MIN);
    self.rho = periodic_map(self.rho, P_RHO.0, P_RHO.1).clamp(-RHO_BOUND, RHO_BOUND);
    self.lambda = periodic_map(self.lambda, P_LAMBDA.0, P_LAMBDA.1).max(0.0);
    self.mu_j = periodic_map(self.mu_j, P_MU_J.0, P_MU_J.1);
    self.sigma_j = periodic_map(self.sigma_j, P_SIGMA_J.0, P_SIGMA_J.1)
      .abs()
      .max(P_SIGMA_J.0);

    if 2.0 * self.kappa * self.theta < self.sigma_v * self.sigma_v {
      let sigma_star = (2.0 * self.kappa * self.theta).sqrt();
      if sigma_star >= P_SIGMA_V.0 {
        self.sigma_v = sigma_star.min(P_SIGMA_V.1);
      } else {
        let theta_star = ((self.sigma_v * self.sigma_v) / (2.0 * self.kappa)).max(THETA_MIN) + EPS;
        self.theta = theta_star.min(P_THETA.1);
      }
    }
  }

  pub fn projected(mut self) -> Self {
    self.project_in_place();
    self
  }
}

impl From<SVJParams> for DVector<f64> {
  fn from(p: SVJParams) -> Self {
    DVector::from_vec(vec![
      p.v0, p.kappa, p.theta, p.sigma_v, p.rho, p.lambda, p.mu_j, p.sigma_j,
    ])
  }
}

impl From<DVector<f64>> for SVJParams {
  fn from(v: DVector<f64>) -> Self {
    SVJParams {
      v0: v[0],
      kappa: v[1],
      theta: v[2],
      sigma_v: v[3],
      rho: v[4],
      lambda: v[5],
      mu_j: v[6],
      sigma_j: v[7],
    }
  }
}
