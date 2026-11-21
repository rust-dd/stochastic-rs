use ndarray::Array1;
use ndarray::ArrayView1;

use super::calibrator::KAPPA_MIN;
use super::calibrator::SIGMA_V_MIN;
use super::calibrator::THETA_MIN;
use crate::calibration::feller_share;
use crate::calibration::feller_share::P_SHARE;
use crate::calibration::least_squares::Room;
use crate::calibration::periodic_map;
use crate::calibration::periodic_slopes;
use crate::calibration::stencil::room;
use crate::pricing::fourier::SvjParams;

pub(super) const EPS: f64 = 1e-8;
pub(super) const RHO_BOUND: f64 = 0.9999;

pub(super) const P_V0: (f64, f64) = (0.005, 0.25);
pub(super) const P_KAPPA: (f64, f64) = (0.1, 20.0);
pub(super) const P_THETA: (f64, f64) = (0.001, 0.4);
pub(super) const P_SIGMA_V: (f64, f64) = (0.01, 1.0);
pub(super) const P_RHO: (f64, f64) = (-RHO_BOUND, RHO_BOUND);
pub(super) const P_LAMBDA: (f64, f64) = (0.0, 10.0);
pub(super) const P_MU_J: (f64, f64) = (-0.5, 0.5);
pub(super) const P_SIGMA_J: (f64, f64) = (0.001, 1.0);
/// The box each raw coordinate folds into, `ρ` at its bound so that no raw step leaves it
/// unmoved; σ's is its share of the way to Feller's cap.
pub(super) const BOX: [(f64, f64); 8] = [
  P_V0, P_KAPPA, P_THETA, P_SHARE, P_RHO, P_LAMBDA, P_MU_J, P_SIGMA_J,
];

/// Each raw coordinate's room before its folded value meets a bound of `BOX`.
pub(super) fn room_in_box(x: ArrayView1<f64>) -> Room {
  let folded = Array1::from(SvjParams::from(x.to_owned()).folded());
  room(&folded, &BOX, periodic_slopes(x, &BOX).view())
}

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

  /// A raw point folded into `BOX`: `sigma` holds σ's share, the fit's coordinate.
  pub(super) fn folded(mut self) -> Self {
    let share = periodic_map(self.sigma, P_SHARE.0, P_SHARE.1);
    let rho = periodic_map(self.rho, P_RHO.0, P_RHO.1);
    self.project_box();
    (self.sigma, self.rho) = (share, rho);
    self
  }

  /// The folded point of a start: its box projection, σ as its share, 1 where it breaks Feller.
  pub(super) fn to_share(mut self) -> Self {
    self.project_box();
    self.sigma = feller_share::share(self.sigma, self.kappa, self.theta, P_SIGMA_V);
    self
  }

  /// The model's values at a folded point, inside Feller's condition at every share.
  pub(super) fn at_share(mut self) -> Self {
    self.sigma = feller_share::sigma(self.sigma, self.kappa, self.theta, P_SIGMA_V);
    self
  }

  /// `[∂σ/∂κ, ∂σ/∂θ, ∂σ/∂w]` at a folded point.
  pub(super) fn share_slopes(&self) -> [f64; 3] {
    feller_share::slopes(self.sigma, self.kappa, self.theta, P_SIGMA_V)
  }

  fn project_box(&mut self) {
    self.v0 = periodic_map(self.v0, P_V0.0, P_V0.1).max(0.0);
    self.kappa = periodic_map(self.kappa, P_KAPPA.0, P_KAPPA.1).max(KAPPA_MIN);
    self.theta = periodic_map(self.theta, P_THETA.0, P_THETA.1).max(THETA_MIN);
    self.sigma = periodic_map(self.sigma, P_SIGMA_V.0, P_SIGMA_V.1)
      .abs()
      .max(SIGMA_V_MIN);
    self.rho = periodic_map(self.rho, -1.0, 1.0).clamp(-RHO_BOUND, RHO_BOUND);
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
