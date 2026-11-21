use ndarray::Array1;
use ndarray::ArrayView1;

use crate::calibration::feller_share;
use crate::calibration::feller_share::P_SHARE;
use crate::calibration::heston::params::EPS;
use crate::calibration::heston::params::P_KAPPA;
use crate::calibration::heston::params::P_RHO;
use crate::calibration::heston::params::P_THETA;
use crate::calibration::heston::params::RHO_BOUND;
use crate::calibration::heston::params::THETA_MIN;
use crate::calibration::least_squares::Room;
use crate::calibration::periodic_map;
use crate::calibration::periodic_slopes;
use crate::calibration::stencil::room;
use crate::pricing::heston::HestonParams;

pub(super) const P_V0: (f64, f64) = (0.001, 0.25);
pub(super) const P_SIGMA: (f64, f64) = (0.01, 1.0);

const FACTOR: [(f64, f64); 5] = [P_V0, P_KAPPA, P_THETA, P_SHARE, (-RHO_BOUND, RHO_BOUND)];

/// Where a folded value lies: each factor's box, with `ρ` clamped inside its reflection's range.
pub(super) const BOXED: [(f64, f64); 10] = [
  FACTOR[0], FACTOR[1], FACTOR[2], FACTOR[3], FACTOR[4], FACTOR[0], FACTOR[1], FACTOR[2],
  FACTOR[3], FACTOR[4],
];

/// The box each raw coordinate is reflected into; σ's is its share of the way to Feller's cap.
pub(super) const FOLD: [(f64, f64); 10] = [
  P_V0, P_KAPPA, P_THETA, P_SHARE, P_RHO, P_V0, P_KAPPA, P_THETA, P_SHARE, P_RHO,
];

/// Each raw coordinate's room before its folded value meets a bound of `BOXED`.
pub(super) fn room_in_box(x: ArrayView1<f64>) -> Room {
  let folded = Array1::from(DoubleHestonParams::from(x.to_owned()).folded());
  room(&folded, &BOXED, periodic_slopes(x, &FOLD).view())
}

/// Double Heston model parameters: two independent Heston factors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoubleHestonParams {
  pub first: HestonParams,
  pub second: HestonParams,
}

impl DoubleHestonParams {
  /// Project parameters to satisfy admissibility: box constraints via periodic
  /// mapping plus per-factor Feller condition $2\kappa_j\theta_j\ge\sigma_j^2$.
  pub fn project_in_place(&mut self) {
    for factor in [&mut self.first, &mut self.second] {
      factor.project_box(P_V0, P_SIGMA);
      project_feller(factor);
    }
  }

  pub fn projected(mut self) -> Self {
    self.project_in_place();
    self
  }

  /// A raw point folded into `FOLD`: each `sigma` holds its σ's share, the fit's coordinate.
  pub(super) fn folded(mut self) -> Self {
    for factor in [&mut self.first, &mut self.second] {
      let share = periodic_map(factor.sigma, P_SHARE.0, P_SHARE.1);
      factor.project_box(P_V0, P_SIGMA);
      factor.sigma = share;
    }
    self
  }

  /// The folded point of a start: its box projection, each σ as its share, 1 where it breaks
  /// Feller.
  pub(super) fn to_share(mut self) -> Self {
    for factor in [&mut self.first, &mut self.second] {
      factor.project_box(P_V0, P_SIGMA);
      factor.sigma = feller_share::share(factor.sigma, factor.kappa, factor.theta, P_SIGMA);
    }
    self
  }

  /// The model's values at a folded point, inside Feller's condition at every share.
  pub(super) fn from_share(mut self) -> Self {
    for factor in [&mut self.first, &mut self.second] {
      factor.sigma = feller_share::sigma(factor.sigma, factor.kappa, factor.theta, P_SIGMA);
    }
    self
  }

  /// Per factor, `[∂σ/∂κ, ∂σ/∂θ, ∂σ/∂w]` at a folded point.
  pub(super) fn share_slopes(&self) -> [[f64; 3]; 2] {
    [self.first, self.second]
      .map(|factor| feller_share::slopes(factor.sigma, factor.kappa, factor.theta, P_SIGMA))
  }
}

fn project_feller(p: &mut HestonParams) {
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
