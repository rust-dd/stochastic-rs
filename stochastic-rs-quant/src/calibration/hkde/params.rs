use nalgebra::DVector;

use super::super::periodic_map;

pub(super) const EPS: f64 = 1e-8;
pub(super) const RHO_BOUND: f64 = 0.9999;
pub(super) const KAPPA_MIN: f64 = 1e-3;
pub(super) const THETA_MIN: f64 = 1e-8;
pub(super) const SIGMA_V_MIN: f64 = 1e-8;
/// Kou admissibility: $\eta_1>1$ is required for $\mathbb E[e^J]$ to be finite.
pub(super) const ETA1_MIN: f64 = 1.0 + 1e-3;
pub(super) const ETA2_MIN: f64 = 1e-3;

/// Box ranges used by the periodic projection.
///
/// The upper bounds were chosen to encompass the calibrated parameter sets
/// reported by Agazzotti et al. (2025), Table 1, which routinely produce
/// $\theta$ values above $0.4$ on single-name equities (e.g. SHOP: $\theta=0.728$).
pub(super) const P_V0: (f64, f64) = (0.005, 0.25);
pub(super) const P_KAPPA: (f64, f64) = (0.1, 20.0);
pub(super) const P_THETA: (f64, f64) = (0.001, 1.0);
pub(super) const P_SIGMA_V: (f64, f64) = (0.01, 1.0);
pub(super) const P_RHO: (f64, f64) = (-1.0, 1.0);
pub(super) const P_LAMBDA: (f64, f64) = (0.0, 10.0);
pub(super) const P_P_UP: (f64, f64) = (0.001, 0.999);
pub(super) const P_ETA1: (f64, f64) = (1.01, 50.0);
pub(super) const P_ETA2: (f64, f64) = (0.1, 50.0);

/// Hkde model parameters — Heston stochastic volatility augmented by a
/// Kou double-exponential jump component.
#[derive(Clone, Copy, Debug)]
pub struct HKDEParams {
  /// Initial variance $v_0$.
  pub v0: f64,
  /// Mean-reversion speed $\kappa$.
  pub kappa: f64,
  /// Long-run variance $\theta$.
  pub theta: f64,
  /// Volatility of variance $\sigma_v$.
  pub sigma_v: f64,
  /// Correlation $\rho$ between the price and variance Brownian motions.
  pub rho: f64,
  /// Poisson jump intensity $\lambda$.
  pub lambda: f64,
  /// Probability $p$ of an upward jump.
  pub p_up: f64,
  /// Upward jump rate $\eta_1>1$.
  pub eta1: f64,
  /// Downward jump rate $\eta_2>0$.
  pub eta2: f64,
}

impl HKDEParams {
  /// Project the parameter vector onto the admissible set.
  ///
  /// The projection enforces the box bounds, the Feller condition
  /// $2\kappa\theta\ge\sigma_v^2$ and the Kou finiteness requirement
  /// $\eta_1>1$.
  pub fn project_in_place(&mut self) {
    self.v0 = periodic_map(self.v0, P_V0.0, P_V0.1).max(0.0);
    self.kappa = periodic_map(self.kappa, P_KAPPA.0, P_KAPPA.1).max(KAPPA_MIN);
    self.theta = periodic_map(self.theta, P_THETA.0, P_THETA.1).max(THETA_MIN);
    self.sigma_v = periodic_map(self.sigma_v, P_SIGMA_V.0, P_SIGMA_V.1)
      .abs()
      .max(SIGMA_V_MIN);
    self.rho = periodic_map(self.rho, P_RHO.0, P_RHO.1).clamp(-RHO_BOUND, RHO_BOUND);
    self.lambda = periodic_map(self.lambda, P_LAMBDA.0, P_LAMBDA.1).max(0.0);
    self.p_up = periodic_map(self.p_up, P_P_UP.0, P_P_UP.1).clamp(P_P_UP.0, P_P_UP.1);
    self.eta1 = periodic_map(self.eta1, P_ETA1.0, P_ETA1.1).max(ETA1_MIN);
    self.eta2 = periodic_map(self.eta2, P_ETA2.0, P_ETA2.1).max(ETA2_MIN);

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

  /// Return a copy with the admissibility projection applied.
  pub fn projected(mut self) -> Self {
    self.project_in_place();
    self
  }
}

impl From<HKDEParams> for DVector<f64> {
  fn from(p: HKDEParams) -> Self {
    DVector::from_vec(vec![
      p.v0, p.kappa, p.theta, p.sigma_v, p.rho, p.lambda, p.p_up, p.eta1, p.eta2,
    ])
  }
}

impl From<DVector<f64>> for HKDEParams {
  fn from(v: DVector<f64>) -> Self {
    HKDEParams {
      v0: v[0],
      kappa: v[1],
      theta: v[2],
      sigma_v: v[3],
      rho: v[4],
      lambda: v[5],
      p_up: v[6],
      eta1: v[7],
      eta2: v[8],
    }
  }
}
