use ndarray::Array1;

use super::calibrator::KAPPA_MIN;
use super::calibrator::SIGMA_V_MIN;
use super::calibrator::THETA_MIN;
use crate::pricing::fourier::SvjParams;

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
pub(super) const BOX: [(f64, f64); 8] = [
  P_V0, P_KAPPA, P_THETA, P_SIGMA_V, P_RHO, P_LAMBDA, P_MU_J, P_SIGMA_J,
];

/// Where a boxed value lies: `BOX`, with `ρ` clamped inside its reflection's range.
pub(super) const BOXED: [(f64, f64); 8] = [
  P_V0,
  P_KAPPA,
  P_THETA,
  P_SIGMA_V,
  (-RHO_BOUND, RHO_BOUND),
  P_LAMBDA,
  P_MU_J,
  P_SIGMA_J,
];

impl SvjParams {
  /// Project parameters to satisfy admissibility constraints (box + Feller condition).
  pub fn project_in_place(&mut self) {
    self.project_box();
    self.project_feller();
  }

  pub fn projected(mut self) -> Self {
    self.project_in_place();
    self
  }

  /// The box alone: Feller's condition, which couples three values, stays for `project_feller`.
  pub(super) fn boxed(mut self) -> Self {
    self.project_box();
    self
  }

  fn project_box(&mut self) {
    use crate::calibration::periodic_map;

    self.v0 = periodic_map(self.v0, P_V0.0, P_V0.1).max(0.0);
    self.kappa = periodic_map(self.kappa, P_KAPPA.0, P_KAPPA.1).max(KAPPA_MIN);
    self.theta = periodic_map(self.theta, P_THETA.0, P_THETA.1).max(THETA_MIN);
    self.sigma = periodic_map(self.sigma, P_SIGMA_V.0, P_SIGMA_V.1)
      .abs()
      .max(SIGMA_V_MIN);
    self.rho = periodic_map(self.rho, P_RHO.0, P_RHO.1).clamp(-RHO_BOUND, RHO_BOUND);
    self.lambda = periodic_map(self.lambda, P_LAMBDA.0, P_LAMBDA.1).max(0.0);
    self.mu_j = periodic_map(self.mu_j, P_MU_J.0, P_MU_J.1);
    self.sigma_j = periodic_map(self.sigma_j, P_SIGMA_J.0, P_SIGMA_J.1)
      .abs()
      .max(P_SIGMA_J.0);
  }

  fn project_feller(&mut self) {
    if 2.0 * self.kappa * self.theta < self.sigma * self.sigma {
      let sigma_star = (2.0 * self.kappa * self.theta).sqrt();
      if sigma_star >= P_SIGMA_V.0 {
        self.sigma = sigma_star.min(P_SIGMA_V.1);
      } else {
        let theta_star = ((self.sigma * self.sigma) / (2.0 * self.kappa)).max(THETA_MIN) + EPS;
        self.theta = theta_star.min(P_THETA.1);
      }
    }
  }
}

impl From<SvjParams> for Array1<f64> {
  fn from(p: SvjParams) -> Self {
    Array1::from_vec(vec![
      p.v0, p.kappa, p.theta, p.sigma, p.rho, p.lambda, p.mu_j, p.sigma_j,
    ])
  }
}

impl From<Array1<f64>> for SvjParams {
  fn from(v: Array1<f64>) -> Self {
    SvjParams {
      v0: v[0],
      kappa: v[1],
      theta: v[2],
      sigma: v[3],
      rho: v[4],
      lambda: v[5],
      mu_j: v[6],
      sigma_j: v[7],
    }
  }
}
