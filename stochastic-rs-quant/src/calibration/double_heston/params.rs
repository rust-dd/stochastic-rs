use nalgebra::DVector;

use super::super::periodic_map;

pub(super) const EPS: f64 = 1e-8;
pub(super) const RHO_BOUND: f64 = 0.9999;
pub(super) const KAPPA_MIN: f64 = 1e-3;
pub(super) const THETA_MIN: f64 = 1e-8;
pub(super) const SIGMA_MIN: f64 = 1e-8;

pub(super) const P_V0: (f64, f64) = (0.001, 0.25);
pub(super) const P_KAPPA: (f64, f64) = (0.1, 20.0);
pub(super) const P_THETA: (f64, f64) = (0.001, 0.4);
pub(super) const P_SIGMA: (f64, f64) = (0.01, 1.0);
pub(super) const P_RHO: (f64, f64) = (-1.0, 1.0);

/// Double Heston model parameters.
#[derive(Clone, Copy, Debug)]
pub struct DoubleHestonParams {
  /// Initial variance of factor 1.
  pub v1_0: f64,
  /// Mean-reversion speed of factor 1.
  pub kappa1: f64,
  /// Long-run variance of factor 1.
  pub theta1: f64,
  /// Volatility-of-variance of factor 1.
  pub sigma1: f64,
  /// Spot-variance correlation for factor 1.
  pub rho1: f64,
  /// Initial variance of factor 2.
  pub v2_0: f64,
  /// Mean-reversion speed of factor 2.
  pub kappa2: f64,
  /// Long-run variance of factor 2.
  pub theta2: f64,
  /// Volatility-of-variance of factor 2.
  pub sigma2: f64,
  /// Spot-variance correlation for factor 2.
  pub rho2: f64,
}

impl DoubleHestonParams {
  /// Project parameters to satisfy admissibility: box constraints via periodic
  /// mapping plus per-factor Feller condition $2\kappa_j\theta_j\ge\sigma_j^2$.
  pub fn project_in_place(&mut self) {
    self.v1_0 = periodic_map(self.v1_0, P_V0.0, P_V0.1).max(0.0);
    self.kappa1 = periodic_map(self.kappa1, P_KAPPA.0, P_KAPPA.1).max(KAPPA_MIN);
    self.theta1 = periodic_map(self.theta1, P_THETA.0, P_THETA.1).max(THETA_MIN);
    self.sigma1 = periodic_map(self.sigma1, P_SIGMA.0, P_SIGMA.1)
      .abs()
      .max(SIGMA_MIN);
    self.rho1 = periodic_map(self.rho1, P_RHO.0, P_RHO.1).clamp(-RHO_BOUND, RHO_BOUND);

    self.v2_0 = periodic_map(self.v2_0, P_V0.0, P_V0.1).max(0.0);
    self.kappa2 = periodic_map(self.kappa2, P_KAPPA.0, P_KAPPA.1).max(KAPPA_MIN);
    self.theta2 = periodic_map(self.theta2, P_THETA.0, P_THETA.1).max(THETA_MIN);
    self.sigma2 = periodic_map(self.sigma2, P_SIGMA.0, P_SIGMA.1)
      .abs()
      .max(SIGMA_MIN);
    self.rho2 = periodic_map(self.rho2, P_RHO.0, P_RHO.1).clamp(-RHO_BOUND, RHO_BOUND);

    // Feller, factor 1.
    if 2.0 * self.kappa1 * self.theta1 < self.sigma1 * self.sigma1 {
      let sigma_star = (2.0 * self.kappa1 * self.theta1).sqrt();
      if sigma_star >= P_SIGMA.0 {
        self.sigma1 = sigma_star.min(P_SIGMA.1);
      } else {
        let theta_star = ((self.sigma1 * self.sigma1) / (2.0 * self.kappa1)).max(THETA_MIN) + EPS;
        self.theta1 = theta_star.min(P_THETA.1);
      }
    }

    // Feller, factor 2.
    if 2.0 * self.kappa2 * self.theta2 < self.sigma2 * self.sigma2 {
      let sigma_star = (2.0 * self.kappa2 * self.theta2).sqrt();
      if sigma_star >= P_SIGMA.0 {
        self.sigma2 = sigma_star.min(P_SIGMA.1);
      } else {
        let theta_star = ((self.sigma2 * self.sigma2) / (2.0 * self.kappa2)).max(THETA_MIN) + EPS;
        self.theta2 = theta_star.min(P_THETA.1);
      }
    }
  }

  pub fn projected(mut self) -> Self {
    self.project_in_place();
    self
  }

  /// Convert to a [`DoubleHestonFourier`] model for pricing / vol surface generation.
  pub fn to_model(&self, r: f64, q: f64) -> crate::pricing::fourier::DoubleHestonFourier {
    crate::pricing::fourier::DoubleHestonFourier {
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
      r,
      q,
    }
  }
}

impl crate::traits::ToModel for DoubleHestonParams {
  type Model = crate::pricing::fourier::DoubleHestonFourier;
  fn to_model(&self, r: f64, q: f64) -> Self::Model {
    DoubleHestonParams::to_model(self, r, q)
  }
}

impl From<DoubleHestonParams> for DVector<f64> {
  fn from(p: DoubleHestonParams) -> Self {
    DVector::from_vec(vec![
      p.v1_0, p.kappa1, p.theta1, p.sigma1, p.rho1, p.v2_0, p.kappa2, p.theta2, p.sigma2, p.rho2,
    ])
  }
}

impl From<DVector<f64>> for DoubleHestonParams {
  fn from(v: DVector<f64>) -> Self {
    DoubleHestonParams {
      v1_0: v[0],
      kappa1: v[1],
      theta1: v[2],
      sigma1: v[3],
      rho1: v[4],
      v2_0: v[5],
      kappa2: v[6],
      theta2: v[7],
      sigma2: v[8],
      rho2: v[9],
    }
  }
}
