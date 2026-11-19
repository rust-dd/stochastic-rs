//! Heston consistent extended Kalman filtering and NMLE parameter estimation.
//!
//! Filter and batch fit are separate entry points; zero returns, a constant price path's, are data.
//!
//! Reference: Wang, He, Bao & Zhao (2018), Sci. China Inf. Sci. 61, DOI: 10.1007/s11432-017-9215-8

use crate::heston_mle::HestonMleFit;

mod batch;
mod filter;

pub use batch::HestonNmleCekfConfig;
pub use batch::HestonNmleCekfCorrection;
pub use batch::HestonNmleCekfFit;
pub use batch::fit_heston_nmle_cekf;
pub use filter::HestonCekfConsistencyBounds;
pub use filter::HestonCekfCorrection;
pub use filter::HestonCekfFilterConfig;
pub use filter::HestonCekfFilterResult;
pub use filter::HestonCekfIndexedVarianceProjection;
pub use filter::HestonCekfPositiveStatePolicy;
pub use filter::HestonCekfProjectionDiagnostics;
pub use filter::HestonCekfState;
pub use filter::HestonCekfStepDiagnostics;
pub use filter::HestonCekfStepResult;
pub use filter::HestonCekfVarianceProjection;
pub use filter::filter_heston_cekf;
pub use filter::heston_cekf_step;

pub(crate) const EPS: f64 = 1e-12;
const BATCH_RHO_MAX: f64 = 0.9999;

/// Heston parameters used by the CEKF state transition and measurement model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HestonNmleCekfParams {
  pub kappa: f64,
  pub theta: f64,
  pub sigma: f64,
  pub rho: f64,
}

impl Default for HestonNmleCekfParams {
  fn default() -> Self {
    Self {
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.4,
      rho: -0.5,
    }
  }
}

impl From<HestonMleFit> for HestonNmleCekfParams {
  fn from(value: HestonMleFit) -> Self {
    Self {
      kappa: value.kappa,
      theta: value.theta,
      sigma: value.sigma,
      rho: value.rho,
    }
  }
}

impl HestonNmleCekfParams {
  /// The batch's bound on an iterate's `rho`; the refit's moment estimate is held in `[-1, 1]`.
  pub(crate) fn projected_batch(mut self) -> Self {
    self.rho = self.rho.clamp(-BATCH_RHO_MAX, BATCH_RHO_MAX);
    self
  }
}

#[cfg(test)]
mod batch_tests;
#[cfg(test)]
mod filter_tests;
#[cfg(test)]
mod test_support;
