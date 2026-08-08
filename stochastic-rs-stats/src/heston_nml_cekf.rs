//! Heston consistent extended Kalman filtering and NMLE parameter estimation.
//!
//! The fixed-parameter filter and the alternating batch estimator are
//! separate APIs. This keeps filtering semantics independent from
//! parameter-update and convergence semantics.
//!
//! Reference: Wang, X., He, X., Bao, Y., Zhao, Y. (2018), "Parameter estimates
//! of Heston stochastic volatility model with MLE and consistent EKF
//! algorithm", *Science China Information Sciences* 61.
//! DOI: 10.1007/s11432-017-9215-8

use std::error::Error;
use std::fmt;

use crate::heston_mle::HestonMleResult;

mod batch;
mod filter;

pub use batch::HestonNMLECEKFConfig;
pub use batch::HestonNMLECEKFResult;
pub use batch::nmle_cekf_heston;
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
pub struct HestonNMLECEKFParams {
  pub kappa: f64,
  pub theta: f64,
  pub sigma: f64,
  pub rho: f64,
}

impl Default for HestonNMLECEKFParams {
  fn default() -> Self {
    Self {
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.4,
      rho: -0.5,
    }
  }
}

impl From<HestonMleResult> for HestonNMLECEKFParams {
  fn from(value: HestonMleResult) -> Self {
    Self {
      kappa: value.kappa,
      theta: value.theta,
      sigma: value.sigma,
      rho: value.rho,
    }
  }
}

impl HestonNMLECEKFParams {
  pub(crate) fn projected_batch(mut self) -> Self {
    self.kappa = self.kappa.abs().max(EPS);
    self.theta = self.theta.abs().max(EPS);
    self.sigma = self.sigma.abs().max(EPS);
    self.rho = self.rho.clamp(-BATCH_RHO_MAX, BATCH_RHO_MAX);
    self
  }
}

/// Validation and filtering errors for the CEKF APIs.
#[derive(Clone, Debug, PartialEq)]
pub enum HestonCekfError {
  TooFewPrices {
    len: usize,
  },
  LengthMismatch {
    prices: usize,
    variances: usize,
  },
  InvalidValue {
    field: &'static str,
    value: f64,
  },
  InvalidPrice {
    index: usize,
    value: f64,
  },
  InvalidVariance {
    index: usize,
    value: f64,
  },
  NonPositiveSemidefiniteNoise,
  ConsistencyBoundViolated {
    coefficient: &'static str,
    actual: f64,
    bound: f64,
  },
  FilterFloorHit {
    quantity: &'static str,
    value: f64,
  },
}

impl fmt::Display for HestonCekfError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::TooFewPrices { len } => write!(f, "at least two prices are required, got {len}"),
      Self::LengthMismatch { prices, variances } => write!(
        f,
        "price and variance lengths must match, got {prices} and {variances}"
      ),
      Self::InvalidValue { field, value } => write!(f, "{field} has invalid value {value}"),
      Self::InvalidPrice { index, value } => {
        write!(
          f,
          "price at index {index} must be finite and positive, got {value}"
        )
      }
      Self::InvalidVariance { index, value } => write!(
        f,
        "variance at index {index} must be finite and positive, got {value}"
      ),
      Self::NonPositiveSemidefiniteNoise => {
        write!(
          f,
          "the process-noise covariance must be positive semidefinite"
        )
      }
      Self::ConsistencyBoundViolated {
        coefficient,
        actual,
        bound,
      } => write!(
        f,
        "consistent-filter bound for {coefficient} is {bound}, below actual magnitude {actual}"
      ),
      Self::FilterFloorHit { quantity, value } => {
        write!(
          f,
          "CEKF {quantity} hit the numerical floor with raw value {value}"
        )
      }
    }
  }
}

impl Error for HestonCekfError {}

#[cfg(test)]
mod batch_tests;
#[cfg(test)]
mod filter_tests;
#[cfg(test)]
mod test_support;
