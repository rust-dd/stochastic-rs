use nalgebra::DVector;
use num_complex::Complex64;
use stochastic_rs_stats::heston_mle::HestonMleResult;

use super::super::periodic_map;

pub(super) const EPS: f64 = 1e-8;
pub(super) const RHO_BOUND: f64 = 0.9999;
pub(super) const KAPPA_MIN: f64 = 1e-3;
pub(super) const THETA_MIN: f64 = 1e-8;
pub(super) const SIGMA_MIN: f64 = 1e-8;

pub(super) const P_KAPPA: (f64, f64) = (0.1, 20.0);
pub(super) const P_THETA: (f64, f64) = (0.001, 0.4);
pub(super) const P_SIGMA: (f64, f64) = (0.01, 0.6);
pub(super) const P_RHO: (f64, f64) = (-1.0, 1.0);
pub(super) const P_V0: (f64, f64) = (0.005, 0.25);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Jacobian strategy used by the Heston least-squares calibration.
///
/// Source:
/// - Cui et al. (2017), analytic Heston calibration Jacobian
///   https://doi.org/10.1016/j.ejor.2017.05.018
pub enum HestonJacobianMethod {
  /// Central finite-difference Jacobian.
  NumericFiniteDiff,
  /// Closed-form analytic Jacobian for Heston calibration integrals.
  ///
  /// Source:
  /// - Cui et al. (2017)
  ///   https://doi.org/10.1016/j.ejor.2017.05.018
  #[default]
  CuiAnalytic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
/// Seed estimators for Heston calibration from historical time series.
///
/// Source:
/// - Wang et al. (2018), NMLE/PMLE/NMLE-CEKF
///   https://doi.org/10.1007/s11432-017-9215-8
///   http://scis.scichina.com/en/2018/042202.pdf
pub enum HestonMleSeedMethod {
  #[default]
  /// Nonlinear MLE based on square-root variance dynamics.
  Nmle,
  /// Pseudo-MLE closed-form approximation.
  Pmle,
  /// NMLE with consistent EKF latent-variance filtering.
  NmleCekf,
}

#[derive(Clone, Debug)]
pub struct HestonParams {
  /// Initial variance v0 (not volatility) in Heston model
  pub v0: f64,
  /// Mean reversion speed
  pub kappa: f64,
  /// Long-run variance
  pub theta: f64,
  /// Volatility of variance
  pub sigma: f64,
  /// Correlation between price and variance Brownian motions
  pub rho: f64,
}

impl HestonParams {
  /// Convert to a [`HestonFourier`] model for pricing / vol surface generation.
  pub fn to_model(&self, r: f64, q: f64) -> crate::pricing::fourier::HestonFourier {
    crate::pricing::fourier::HestonFourier {
      v0: self.v0,
      kappa: self.kappa,
      theta: self.theta,
      sigma: self.sigma,
      rho: self.rho,
      r,
      q,
    }
  }
}

impl crate::traits::ToModel for HestonParams {
  type Model = crate::pricing::fourier::HestonFourier;
  fn to_model(&self, r: f64, q: f64) -> Self::Model {
    HestonParams::to_model(self, r, q)
  }
}

impl HestonParams {
  /// Project parameters to satisfy Heston admissibility constraints and periodic-range mapping.
  /// Steps:
  /// 1) Periodic mapping into fixed parameter ranges
  /// 2) Enforce basic positivity/box constraints
  /// 3) Enforce Feller by lowering sigma when needed (otherwise minimally bump theta)
  ///
  /// Source:
  /// - Heston (1993), variance process admissibility context
  ///   https://doi.org/10.1093/rfs/6.2.327
  pub fn project_in_place(&mut self) {
    self.kappa = periodic_map(self.kappa, P_KAPPA.0, P_KAPPA.1);
    self.theta = periodic_map(self.theta, P_THETA.0, P_THETA.1);
    self.sigma = periodic_map(self.sigma, P_SIGMA.0, P_SIGMA.1).abs();
    self.rho = periodic_map(self.rho, P_RHO.0, P_RHO.1);
    self.v0 = periodic_map(self.v0, P_V0.0, P_V0.1);

    self.v0 = self.v0.max(0.0);
    self.kappa = self.kappa.max(KAPPA_MIN);
    self.theta = self.theta.max(THETA_MIN);
    self.sigma = self.sigma.abs().max(SIGMA_MIN);
    self.rho = self.rho.clamp(-RHO_BOUND, RHO_BOUND);

    if 2.0 * self.kappa * self.theta < self.sigma * self.sigma {
      let sigma_star = (2.0 * self.kappa * self.theta).sqrt();
      if sigma_star >= P_SIGMA.0 {
        self.sigma = sigma_star.min(P_SIGMA.1);
      } else {
        let theta_star = ((self.sigma * self.sigma) / (2.0 * self.kappa)).max(THETA_MIN) + EPS;
        self.theta = theta_star.min(P_THETA.1);
      }
    }
  }

  pub fn projected(mut self) -> Self {
    self.project_in_place();
    self
  }
}

impl From<HestonMleResult> for HestonParams {
  fn from(mle: HestonMleResult) -> Self {
    HestonParams {
      v0: mle.v0,
      kappa: mle.kappa,
      theta: mle.theta,
      sigma: mle.sigma,
      rho: mle.rho,
    }
  }
}

impl From<HestonParams> for DVector<f64> {
  fn from(params: HestonParams) -> Self {
    DVector::from_vec(vec![
      params.v0,
      params.kappa,
      params.theta,
      params.sigma,
      params.rho,
    ])
  }
}

impl From<DVector<f64>> for HestonParams {
  fn from(params: DVector<f64>) -> Self {
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
