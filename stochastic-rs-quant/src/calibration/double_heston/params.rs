use ndarray::Array1;

use super::super::periodic_map;
use crate::pricing::heston::HestonParams;

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

/// Double Heston model parameters: two independent Heston factors.
#[derive(Clone, Copy, Debug)]
pub struct DoubleHestonParams {
  pub first: HestonParams,
  pub second: HestonParams,
}

impl DoubleHestonParams {
  /// Project parameters to satisfy admissibility: box constraints via periodic
  /// mapping plus per-factor Feller condition $2\kappa_j\theta_j\ge\sigma_j^2$.
  pub fn project_in_place(&mut self) {
    project_factor(&mut self.first);
    project_factor(&mut self.second);
  }

  pub fn projected(mut self) -> Self {
    self.project_in_place();
    self
  }
}

fn project_factor(p: &mut HestonParams) {
  p.v0 = periodic_map(p.v0, P_V0.0, P_V0.1).max(0.0);
  p.kappa = periodic_map(p.kappa, P_KAPPA.0, P_KAPPA.1).max(KAPPA_MIN);
  p.theta = periodic_map(p.theta, P_THETA.0, P_THETA.1).max(THETA_MIN);
  p.sigma = periodic_map(p.sigma, P_SIGMA.0, P_SIGMA.1)
    .abs()
    .max(SIGMA_MIN);
  p.rho = periodic_map(p.rho, P_RHO.0, P_RHO.1).clamp(-RHO_BOUND, RHO_BOUND);

  if 2.0 * p.kappa * p.theta < p.sigma * p.sigma {
    let sigma_star = (2.0 * p.kappa * p.theta).sqrt();
    if sigma_star >= P_SIGMA.0 {
      p.sigma = sigma_star.min(P_SIGMA.1);
    } else {
      let theta_star = ((p.sigma * p.sigma) / (2.0 * p.kappa)).max(THETA_MIN) + EPS;
      p.theta = theta_star.min(P_THETA.1);
    }
  }
}

impl crate::traits::ToModel for DoubleHestonParams {
  type Model = crate::pricing::fourier::DoubleHestonFourier;
  fn to_model(&self) -> Self::Model {
    crate::pricing::fourier::DoubleHestonFourier::new(self.first, self.second)
  }
}

impl From<DoubleHestonParams> for Array1<f64> {
  fn from(p: DoubleHestonParams) -> Self {
    let (a, b) = (p.first, p.second);
    Array1::from_vec(vec![
      a.v0, a.kappa, a.theta, a.sigma, a.rho, b.v0, b.kappa, b.theta, b.sigma, b.rho,
    ])
  }
}

impl From<Array1<f64>> for DoubleHestonParams {
  fn from(v: Array1<f64>) -> Self {
    let factor = |at: usize| HestonParams {
      v0: v[at],
      kappa: v[at + 1],
      theta: v[at + 2],
      sigma: v[at + 3],
      rho: v[at + 4],
    };
    DoubleHestonParams {
      first: factor(0),
      second: factor(5),
    }
  }
}
