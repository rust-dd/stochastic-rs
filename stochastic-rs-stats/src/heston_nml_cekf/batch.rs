//! Backward-compatible alternating batch refinement API.

use ndarray::Array1;
use ndarray::ArrayView1;

use super::EPS;
use super::HestonNMLECEKFParams;
use super::filter::HestonCekfConsistencyBounds;
use super::filter::HestonCekfCorrection;
use super::filter::HestonCekfFilterConfig;
use super::filter::HestonCekfPositiveStatePolicy;
use super::filter::filter_heston_cekf_batch;
use crate::heston_mle::HestonMleResult;
use crate::heston_mle::nmle_heston_with_delta;

/// Configuration of the alternating batch estimator (published API).
#[derive(Clone, Debug)]
pub struct HestonNMLECEKFConfig {
  pub r: f64,
  pub delta: f64,
  pub max_iters: usize,
  pub tol: f64,
  pub param_damping: f64,
  pub initial_v0: f64,
  pub initial_p0: f64,
  pub initial_params: HestonNMLECEKFParams,
  pub q11: f64,
  pub q12: f64,
  pub q22: f64,
  /// Enables the original CEKF correction terms using current coefficients as
  /// surrogate bounds.
  ///
  /// This compatibility mode does not provide the prior-bound consistency
  /// guarantee of `HestonCekfCorrection::Consistent` in the new filter API.
  pub use_consistent_terms: bool,
}

impl Default for HestonNMLECEKFConfig {
  fn default() -> Self {
    Self {
      r: 0.0,
      delta: 1.0 / 252.0,
      max_iters: 12,
      tol: 1e-6,
      param_damping: 0.7,
      initial_v0: 0.04,
      initial_p0: 0.1,
      initial_params: HestonNMLECEKFParams::default(),
      q11: 1.0,
      q12: 0.0,
      q22: 1.0,
      use_consistent_terms: true,
    }
  }
}

/// Output of the alternating batch estimator (published API).
#[derive(Clone, Debug)]
pub struct HestonNMLECEKFResult {
  pub params: HestonMleResult,
  pub vol_path: Array1<f64>,
  pub cov_path: Array1<f64>,
  pub iterations: usize,
  pub converged: bool,
}

/// Runs the damped, alternating full-batch fixed-point heuristic.
///
/// Each iteration filters the entire series and then refreshes every Heston
/// parameter. `converged` refers only to that batch parameter loop; it is not a
/// CEKF filtering property and this composition is not the causal online
/// recursion from Wang et al. After termination, no additional parameter
/// refresh is performed.
pub fn nmle_cekf_heston(
  prices: ArrayView1<'_, f64>,
  config: HestonNMLECEKFConfig,
) -> HestonNMLECEKFResult {
  validate_batch_config(prices, &config);
  let mut parameters = config.initial_params.projected_batch();
  let mut converged = false;
  let mut iterations = 0;

  for iteration in 0..config.max_iters {
    let (variance_path, _) = cekf_pass(prices, parameters, &config);
    let nmle = nmle_heston_with_delta(prices, variance_path.view(), config.r, config.delta);
    let updated = HestonNMLECEKFParams::from(nmle).projected_batch();
    let blended = blend_params(parameters, updated, config.param_damping).projected_batch();
    let max_difference = (blended.kappa - parameters.kappa)
      .abs()
      .max((blended.theta - parameters.theta).abs())
      .max((blended.sigma - parameters.sigma).abs())
      .max((blended.rho - parameters.rho).abs());

    parameters = blended;
    iterations = iteration + 1;
    if max_difference < config.tol {
      converged = true;
      break;
    }
  }

  let (variance_path, covariance_path) = cekf_pass(prices, parameters, &config);
  HestonNMLECEKFResult {
    params: HestonMleResult {
      v0: variance_path[0].max(0.0),
      kappa: parameters.kappa,
      theta: parameters.theta,
      sigma: parameters.sigma,
      rho: parameters.rho,
    },
    vol_path: variance_path,
    cov_path: covariance_path,
    iterations,
    converged,
  }
}

pub(super) fn cekf_pass(
  prices: ArrayView1<'_, f64>,
  parameters: HestonNMLECEKFParams,
  config: &HestonNMLECEKFConfig,
) -> (Array1<f64>, Array1<f64>) {
  let filter_config = HestonCekfFilterConfig {
    r: config.r,
    delta: config.delta,
    initial_variance: config.initial_v0,
    initial_error_covariance_bound: config.initial_p0,
    q11: config.q11,
    q12: config.q12,
    q22: config.q22,
    correction: batch_correction(parameters, config),
    positive_state_policy: HestonCekfPositiveStatePolicy::Strict,
  };
  let result = filter_heston_cekf_batch(prices, parameters, &filter_config);
  (result.variance_path, result.error_covariance_bound_path)
}

fn batch_correction(
  parameters: HestonNMLECEKFParams,
  config: &HestonNMLECEKFConfig,
) -> HestonCekfCorrection {
  if config.use_consistent_terms {
    HestonCekfCorrection::Consistent {
      bounds: HestonCekfConsistencyBounds {
        max_abs_state_transition: (1.0 - parameters.kappa * config.delta).abs(),
        max_abs_kappa_theta: (parameters.kappa * parameters.theta).abs(),
        max_abs_sigma: parameters.sigma.abs(),
      },
    }
  } else {
    HestonCekfCorrection::Traditional
  }
}

fn blend_params(
  old: HestonNMLECEKFParams,
  new: HestonNMLECEKFParams,
  alpha: f64,
) -> HestonNMLECEKFParams {
  let weight = alpha.clamp(0.0, 1.0);
  HestonNMLECEKFParams {
    kappa: (1.0 - weight) * old.kappa + weight * new.kappa,
    theta: (1.0 - weight) * old.theta + weight * new.theta,
    sigma: (1.0 - weight) * old.sigma + weight * new.sigma,
    rho: (1.0 - weight) * old.rho + weight * new.rho,
  }
}

fn validate_batch_config(prices: ArrayView1<'_, f64>, config: &HestonNMLECEKFConfig) {
  assert!(
    prices.len() >= 2,
    "nmle_cekf_heston requires at least 2 prices"
  );
  assert!(
    config.delta.is_finite() && config.delta > 0.0,
    "delta must be finite and positive"
  );
  assert!(config.max_iters > 0, "max_iters must be positive");
  assert!(
    config.tol.is_finite() && config.tol > 0.0,
    "tol must be positive"
  );
  assert!(
    config.initial_v0.is_finite() && config.initial_v0 > 0.0,
    "initial_v0 must be positive"
  );
  assert!(
    config.initial_p0.is_finite() && config.initial_p0 > 0.0,
    "initial_p0 must be positive"
  );
  assert!(
    config.q11 >= 0.0 && config.q22 >= 0.0,
    "q11 and q22 must be non-negative"
  );
  assert!(
    config.q11 * config.q22 - config.q12 * config.q12 >= -EPS,
    "noise covariance must be positive semidefinite"
  );
}
