use ndarray::Array1;
use num_complex::Complex64;
use stochastic_rs_stats::heston_mle::HestonMleFit;

use super::super::periodic_map;
use crate::pricing::heston::HestonParams;

pub(crate) const EPS: f64 = 1e-8;
pub(crate) const RHO_BOUND: f64 = 0.9999;
pub(crate) const KAPPA_MIN: f64 = 1e-3;
pub(crate) const THETA_MIN: f64 = 1e-8;
pub(crate) const SIGMA_MIN: f64 = 1e-8;

pub(crate) const P_KAPPA: (f64, f64) = (0.1, 20.0);
pub(crate) const P_THETA: (f64, f64) = (0.001, 0.4);
pub(super) const P_SIGMA: (f64, f64) = (0.01, 3.0);
pub(crate) const P_RHO: (f64, f64) = (-1.0, 1.0);
pub(super) const P_V0: (f64, f64) = (0.005, 0.25);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Jacobian strategy of the Heston least-squares calibration.
/// Reference: Cui et al. (2017), Eur. J. Oper. Res. 263(2), 625–638, DOI 10.1016/j.ejor.2017.05.018
#[non_exhaustive]
pub enum HestonJacobianMethod {
  /// Central finite-difference Jacobian.
  NumericFiniteDiff,
  /// Closed-form analytic Jacobian of the calibration integrals, after Cui et al. (2017).
  #[default]
  CuiAnalytic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
/// Seed estimators for Heston calibration from historical time series (NMLE, PMLE, NMLE-CEKF).
/// Reference: Wang et al. (2018), Sci. China Inf. Sci. 61, 042202, DOI 10.1007/s11432-017-9215-8
/// Source: <http://scis.scichina.com/en/2018/042202.pdf>
#[non_exhaustive]
pub enum HestonMleSeedMethod {
  #[default]
  /// Nonlinear MLE based on square-root variance dynamics.
  Nmle,
  /// Pseudo-MLE closed-form approximation.
  Pmle,
  /// NMLE with consistent EKF latent-variance filtering.
  NmleCekf,
}

impl crate::traits::ToModel for HestonParams {
  type Model = crate::pricing::fourier::HestonFourier;
  fn to_model(&self) -> Self::Model {
    crate::pricing::fourier::HestonFourier::new(*self)
  }
}

impl HestonParams {
  /// Project into the calibration box without Feller's condition, which Fourier pricing does not
  /// need and market fits often violate.
  pub fn project_in_place(&mut self) {
    self.project_box(P_V0, P_SIGMA);
  }

  /// Wrap each value into its range, `v0` and `sigma` into the ones given, and floor it at its
  /// minimum; the double-Heston factors share it with their own two ranges.
  pub(crate) fn project_box(&mut self, v0: (f64, f64), sigma: (f64, f64)) {
    self.v0 = periodic_map(self.v0, v0.0, v0.1).max(0.0);
    self.kappa = periodic_map(self.kappa, P_KAPPA.0, P_KAPPA.1).max(KAPPA_MIN);
    self.theta = periodic_map(self.theta, P_THETA.0, P_THETA.1).max(THETA_MIN);
    self.sigma = periodic_map(self.sigma, sigma.0, sigma.1)
      .abs()
      .max(SIGMA_MIN);
    self.rho = periodic_map(self.rho, P_RHO.0, P_RHO.1).clamp(-RHO_BOUND, RHO_BOUND);
  }

  /// Whether the CIR variance process stays strictly positive under Feller's condition.
  pub fn satisfies_feller_condition(&self) -> bool {
    2.0 * self.kappa * self.theta >= self.sigma * self.sigma
  }

  /// Box-projects, then lowers σ alone to the largest [`Self::satisfies_feller_condition`] accepts;
  /// idempotent, and the box keeps √(2κθ) ≥ √(2·0.1·0.001) above σ's floor of 0.01.
  pub fn project_feller_in_place(&mut self) {
    self.project_in_place();
    self.sigma = self.sigma.min(feller_sigma_cap(self.kappa, self.theta));
  }

  pub fn projected(mut self) -> Self {
    self.project_in_place();
    self
  }

  /// Return a box-projected parameter set that also satisfies Feller's condition.
  pub fn projected_with_feller_condition(mut self) -> Self {
    self.project_feller_in_place();
    self
  }
}

/// The largest σ with `σ * σ <= 2.0 * kappa * theta` in `f64`; `√(2κθ)` itself can square one ulp
/// past the bound.
pub(crate) fn feller_sigma_cap(kappa: f64, theta: f64) -> f64 {
  let bound = 2.0 * kappa * theta;
  let mut cap = bound.sqrt();
  while cap * cap > bound {
    cap = cap.next_down();
  }
  cap
}

impl From<HestonMleFit> for HestonParams {
  fn from(mle: HestonMleFit) -> Self {
    HestonParams {
      v0: mle.v0,
      kappa: mle.kappa,
      theta: mle.theta,
      sigma: mle.sigma,
      rho: mle.rho,
    }
  }
}

impl From<HestonParams> for Array1<f64> {
  fn from(params: HestonParams) -> Self {
    Array1::from_vec(vec![
      params.v0,
      params.kappa,
      params.theta,
      params.sigma,
      params.rho,
    ])
  }
}

impl From<Array1<f64>> for HestonParams {
  fn from(params: Array1<f64>) -> Self {
    HestonParams {
      v0: params[0],
      kappa: params[1],
      theta: params[2],
      sigma: params[3],
      rho: params[4],
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct CuiCfTerms {
  pub iu: Complex64,
  pub u2_iu: Complex64,
  pub xi: Complex64,
  pub d: Complex64,
  pub sinh_z: Complex64,
  pub cosh_z: Complex64,
  pub a2: Complex64,
  pub a: Complex64,
  pub b: Complex64,
  pub dlog: Complex64,
  pub phi: Complex64,
  pub exp_half_kappa_t: f64,
}

pub(super) fn finite_c64(z: Complex64) -> bool {
  z.re.is_finite() && z.im.is_finite()
}
