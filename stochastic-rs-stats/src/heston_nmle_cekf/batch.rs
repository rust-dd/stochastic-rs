//! The damped, alternating batch estimator: filter the series, refresh the parameters, repeat.

use ndarray::Array1;
use ndarray::ArrayView1;

use super::EPS;
use super::HestonNmleCekfParams;
use super::filter::HestonCekfConsistencyBounds;
use super::filter::HestonCekfCorrection;
use super::filter::HestonCekfFilterConfig;
use super::filter::HestonCekfPositiveStatePolicy;
use super::filter::assert_finite;
use super::filter::assert_noise;
use super::filter::assert_nonnegative;
use super::filter::assert_positive;
use super::filter::filter_heston_cekf_batch;
use super::filter::validate_parameters;
use super::filter::validate_prices;
use crate::StatsError;
use crate::heston_mle::HestonMleFit;
use crate::heston_mle::fit_heston_nmle_with_delta;

/// Configuration of the alternating batch estimator (published API).
#[derive(Clone, Debug)]
pub struct HestonNmleCekfConfig {
  pub r: f64,
  pub delta: f64,
  pub max_iters: usize,
  pub tol: f64,
  pub param_damping: f64,
  pub initial_v0: f64,
  pub initial_p0: f64,
  pub initial_params: HestonNmleCekfParams,
  pub q11: f64,
  pub q12: f64,
  pub q22: f64,
  /// How the covariance step is corrected; the default rebuilds the bounds from each iterate.
  pub correction: HestonNmleCekfCorrection,
}

/// How the batch estimator corrects the EKF covariance step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum HestonNmleCekfCorrection {
  /// The plain EKF covariance equations.
  Traditional,
  /// Consistent-EKF terms bounded by each iterate's own coefficients: no prior-bound guarantee.
  #[default]
  IterateBounds,
  /// Consistent-EKF terms against prior bounds over the admissible set: initial parameters outside
  /// them panic, and a refreshed iterate outside them is `Err(Numerical)` at "consistency bounds".
  PriorBounds(HestonCekfConsistencyBounds),
}

impl Default for HestonNmleCekfConfig {
  fn default() -> Self {
    Self {
      r: 0.0,
      delta: 1.0 / 252.0,
      max_iters: 12,
      tol: 1e-6,
      param_damping: 0.7,
      initial_v0: 0.04,
      initial_p0: 0.1,
      initial_params: HestonNmleCekfParams::default(),
      q11: 1.0,
      q12: 0.0,
      q22: 1.0,
      correction: HestonNmleCekfCorrection::IterateBounds,
    }
  }
}

/// Output of the alternating batch estimator (published API).
#[derive(Clone, Debug)]
pub struct HestonNmleCekfFit {
  pub params: HestonMleFit,
  pub vol_path: Array1<f64>,
  pub cov_path: Array1<f64>,
  pub iterations: usize,
  pub converged: bool,
}

/// Damped CEKF/NMLE fixed point (`converged` is that loop's); config panics; `Err` below 2 prices,
/// on a non-finite or non-positive one, a non-finite step, a failed refit or a `PriorBounds` exit.
pub fn fit_heston_nmle_cekf(
  prices: ArrayView1<'_, f64>,
  config: HestonNmleCekfConfig,
) -> Result<HestonNmleCekfFit, StatsError> {
  validate_batch_config(&config);
  validate_prices(prices)?;
  let mut parameters = config.initial_params.projected_batch();
  let mut converged = false;
  let mut iterations = 0;

  for iteration in 0..config.max_iters {
    let (variance_path, _) = cekf_pass(prices, parameters, &config)?;
    let nmle = fit_heston_nmle_with_delta(prices, variance_path.view(), config.r, config.delta)
      .map_err(|_| StatsError::Numerical {
        stage: "batch variance path",
      })?;
    let updated = HestonNmleCekfParams::from(nmle).projected_batch();
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

  let (variance_path, covariance_path) = cekf_pass(prices, parameters, &config)?;
  Ok(HestonNmleCekfFit {
    params: HestonMleFit {
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
  })
}

pub(super) fn cekf_pass(
  prices: ArrayView1<'_, f64>,
  parameters: HestonNmleCekfParams,
  config: &HestonNmleCekfConfig,
) -> Result<(Array1<f64>, Array1<f64>), StatsError> {
  if let HestonNmleCekfCorrection::PriorBounds(bounds) = config.correction
    && !within(coefficients(parameters, config.delta), bounds)
  {
    return Err(StatsError::Numerical {
      stage: "consistency bounds",
    });
  }
  let result = filter_heston_cekf_batch(prices, parameters, &filter_config(parameters, config))?;
  Ok((result.variance_path, result.error_covariance_bound_path))
}

fn filter_config(
  parameters: HestonNmleCekfParams,
  config: &HestonNmleCekfConfig,
) -> HestonCekfFilterConfig {
  HestonCekfFilterConfig {
    r: config.r,
    delta: config.delta,
    initial_variance: config.initial_v0,
    initial_error_covariance_bound: config.initial_p0,
    q11: config.q11,
    q12: config.q12,
    q22: config.q22,
    correction: batch_correction(parameters, config),
    positive_state_policy: HestonCekfPositiveStatePolicy::Strict,
  }
}

fn batch_correction(
  parameters: HestonNmleCekfParams,
  config: &HestonNmleCekfConfig,
) -> HestonCekfCorrection {
  match config.correction {
    HestonNmleCekfCorrection::Traditional => HestonCekfCorrection::Traditional,
    HestonNmleCekfCorrection::IterateBounds => HestonCekfCorrection::Consistent {
      bounds: coefficients(parameters, config.delta),
    },
    HestonNmleCekfCorrection::PriorBounds(bounds) => HestonCekfCorrection::Consistent { bounds },
  }
}

/// The iterate's own `|1 − κΔ|`, `|κθ|` and `|σ|`, the coefficients the consistent terms bound.
fn coefficients(parameters: HestonNmleCekfParams, delta: f64) -> HestonCekfConsistencyBounds {
  HestonCekfConsistencyBounds {
    max_abs_state_transition: (1.0 - parameters.kappa * delta).abs(),
    max_abs_kappa_theta: (parameters.kappa * parameters.theta).abs(),
    max_abs_sigma: parameters.sigma.abs(),
  }
}

/// Every coefficient within its prior bound, with the filter's `EPS` slack; false on a NaN.
fn within(coefficients: HestonCekfConsistencyBounds, prior: HestonCekfConsistencyBounds) -> bool {
  coefficients.max_abs_state_transition <= prior.max_abs_state_transition + EPS
    && coefficients.max_abs_kappa_theta <= prior.max_abs_kappa_theta + EPS
    && coefficients.max_abs_sigma <= prior.max_abs_sigma + EPS
}

fn blend_params(
  old: HestonNmleCekfParams,
  new: HestonNmleCekfParams,
  alpha: f64,
) -> HestonNmleCekfParams {
  let weight = alpha.clamp(0.0, 1.0);
  HestonNmleCekfParams {
    kappa: (1.0 - weight) * old.kappa + weight * new.kappa,
    theta: (1.0 - weight) * old.theta + weight * new.theta,
    sigma: (1.0 - weight) * old.sigma + weight * new.sigma,
    rho: (1.0 - weight) * old.rho + weight * new.rho,
  }
}

fn validate_batch_config(config: &HestonNmleCekfConfig) {
  assert_finite("r", config.r);
  assert_positive("delta", config.delta);
  assert!(
    config.max_iters > 0,
    "max_iters must satisfy `max_iters > 0`, got max_iters = {:?}",
    config.max_iters
  );
  assert_positive("tol", config.tol);
  assert_finite("param_damping", config.param_damping);
  assert_positive("initial_v0", config.initial_v0);
  assert_positive("initial_p0", config.initial_p0);
  assert_finite("initial_params.kappa", config.initial_params.kappa);
  assert_finite("initial_params.theta", config.initial_params.theta);
  assert_finite("initial_params.sigma", config.initial_params.sigma);
  assert_finite("initial_params.rho", config.initial_params.rho);
  assert_noise(config.q11, config.q12, config.q22);
  if let HestonNmleCekfCorrection::PriorBounds(bounds) = config.correction {
    assert_nonnegative("max_abs_state_transition", bounds.max_abs_state_transition);
    assert_nonnegative("max_abs_kappa_theta", bounds.max_abs_kappa_theta);
    assert_nonnegative("max_abs_sigma", bounds.max_abs_sigma);
    let initial = config.initial_params.projected_batch();
    validate_parameters(initial, &filter_config(initial, config));
  }
}
