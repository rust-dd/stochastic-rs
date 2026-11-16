use std::panic::UnwindSafe;
use std::panic::catch_unwind;

use super::HestonCekfConsistencyBounds;
use super::HestonCekfCorrection;
use super::HestonCekfFilterConfig;
use super::HestonNmleCekfConfig;
use super::HestonNmleCekfParams;
use super::batch::cekf_pass;
use super::filter::filter_heston_cekf_batch;
use super::filter_heston_cekf;
use super::fit_heston_nmle_cekf;
use super::test_support::simulate_heston_prices;
use crate::StatsError;

fn panic_text(f: impl FnOnce() + UnwindSafe) -> String {
  let payload = catch_unwind(f).expect_err("the call accepted an invalid configuration");
  payload
    .downcast_ref::<String>()
    .cloned()
    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
    .expect("a text panic payload")
}

#[test]
fn batch_estimator_returns_finite_outputs() {
  let observations = 320;
  let delta = 1.0 / (observations as f64 - 1.0);
  let (prices, _) = simulate_heston_prices(observations, delta, 42);
  let config = HestonNmleCekfConfig {
    r: 0.01,
    delta,
    max_iters: 8,
    tol: 1e-5,
    param_damping: 0.6,
    initial_v0: 0.06,
    initial_p0: 0.2,
    initial_params: HestonNmleCekfParams {
      kappa: 1.0,
      theta: 0.03,
      sigma: 0.4,
      rho: -0.3,
    },
    ..HestonNmleCekfConfig::default()
  };
  let result = fit_heston_nmle_cekf(prices.view(), config).unwrap();

  assert_eq!(result.vol_path.len(), observations);
  assert_eq!(result.cov_path.len(), observations);
  assert!(result.iterations >= 1);
  assert!(
    result
      .vol_path
      .iter()
      .all(|value| value.is_finite() && *value > 0.0)
  );
  assert!(
    result
      .cov_path
      .iter()
      .all(|value| value.is_finite() && *value > 0.0)
  );
  assert!(result.params.kappa.is_finite() && result.params.kappa > 0.0);
  assert!(result.params.theta.is_finite() && result.params.theta > 0.0);
  assert!(result.params.sigma.is_finite() && result.params.sigma > 0.0);
  assert!(result.params.rho.is_finite() && result.params.rho.abs() <= 1.0);
}

#[test]
fn batch_reported_parameters_reproduce_the_returned_path() {
  let delta = 1.0 / 252.0;
  let (prices, _) = simulate_heston_prices(360, delta, 53);
  let config = HestonNmleCekfConfig {
    r: 0.015,
    delta,
    max_iters: 2,
    tol: 1e-12,
    param_damping: 0.35,
    initial_v0: 0.07,
    ..HestonNmleCekfConfig::default()
  };
  let result = fit_heston_nmle_cekf(prices.view(), config.clone()).unwrap();
  let reported = HestonNmleCekfParams::from(result.params.clone()).projected_batch();
  let (expected_variance, expected_covariance) =
    cekf_pass(prices.view(), reported, &config).unwrap();

  assert_eq!(result.vol_path, expected_variance);
  assert_eq!(result.cov_path, expected_covariance);
}

#[test]
fn batch_stops_without_a_hidden_parameter_refresh() {
  let delta = 1.0 / 252.0;
  let (prices, _) = simulate_heston_prices(256, delta, 59);
  let initial = HestonNmleCekfParams {
    kappa: 0.91,
    theta: 0.071,
    sigma: 0.47,
    rho: -0.28,
  };
  let config = HestonNmleCekfConfig {
    delta,
    max_iters: 1,
    tol: 1e-15,
    param_damping: 0.0,
    initial_params: initial,
    ..HestonNmleCekfConfig::default()
  };
  let result = fit_heston_nmle_cekf(prices.view(), config).unwrap();

  assert_eq!(result.params.kappa.to_bits(), initial.kappa.to_bits());
  assert_eq!(result.params.theta.to_bits(), initial.theta.to_bits());
  assert_eq!(result.params.sigma.to_bits(), initial.sigma.to_bits());
  assert_eq!(result.params.rho.to_bits(), initial.rho.to_bits());
}

#[test]
fn batch_nonidentity_noise_uses_the_corrected_filter_core() {
  let delta = 1.0 / 252.0;
  let (prices, _) = simulate_heston_prices(64, delta, 61);
  let parameters = HestonNmleCekfParams {
    kappa: 1.1,
    theta: 0.05,
    sigma: 0.36,
    rho: -0.4,
  };
  let batch = HestonNmleCekfConfig {
    delta,
    initial_params: parameters,
    initial_p0: 0.0004,
    q11: 1.4,
    q12: 0.35,
    q22: 1.2,
    use_consistent_terms: true,
    ..HestonNmleCekfConfig::default()
  };
  let (batch_variance, batch_covariance) = cekf_pass(prices.view(), parameters, &batch).unwrap();
  let filter = HestonCekfFilterConfig {
    r: batch.r,
    delta,
    initial_variance: batch.initial_v0,
    initial_error_covariance_bound: batch.initial_p0,
    q11: batch.q11,
    q12: batch.q12,
    q22: batch.q22,
    correction: HestonCekfCorrection::Consistent {
      bounds: HestonCekfConsistencyBounds {
        max_abs_state_transition: (1.0 - parameters.kappa * delta).abs(),
        max_abs_kappa_theta: parameters.kappa * parameters.theta,
        max_abs_sigma: parameters.sigma,
      },
    },
    positive_state_policy: super::HestonCekfPositiveStatePolicy::Strict,
  };
  let corrected = filter_heston_cekf(prices.view(), parameters, &filter).unwrap();

  assert_eq!(batch_variance, corrected.variance_path);
  assert_eq!(batch_covariance, corrected.error_covariance_bound_path);
}

#[test]
fn the_batch_rejects_bad_prices_before_filtering() {
  let (mut prices, _) = simulate_heston_prices(64, 1.0 / 252.0, 67);
  assert_eq!(
    fit_heston_nmle_cekf(
      prices.slice(ndarray::s![..1]),
      HestonNmleCekfConfig::default()
    )
    .unwrap_err(),
    crate::StatsError::InsufficientData { needed: 2, got: 1 }
  );
  prices[9] = 0.0;
  assert_eq!(
    fit_heston_nmle_cekf(prices.view(), HestonNmleCekfConfig::default()).unwrap_err(),
    crate::StatsError::InvalidObservation {
      index: 9,
      value: 0.0,
      constraint: "finite and > 0"
    }
  );
}

/// The default correction reproduces the pinned fit bit for bit.
#[test]
fn the_default_correction_reproduces_the_pinned_fit_bit_for_bit() {
  for (observations, seed, pinned, variance) in [
    (
      64,
      61,
      [
        0.04,
        25.022489555104894,
        0.04326240779613097,
        2.0568885162403605,
        -0.9998997343326441,
      ],
      [1e-12, 1e-12, 1e-12],
    ),
    (
      128,
      7,
      [
        0.04,
        196.68353068468784,
        0.01558564991592691,
        3.4982296798301187,
        -0.9998997343326441,
      ],
      [1e-12, 0.01217715390796855, 0.05835715408131442],
    ),
  ] {
    let (prices, _) = simulate_heston_prices(observations, 1.0 / 252.0, seed);
    let fit = fit_heston_nmle_cekf(prices.view(), HestonNmleCekfConfig::default()).unwrap();
    let params = fit.params;
    let got = [
      params.v0,
      params.kappa,
      params.theta,
      params.sigma,
      params.rho,
    ];
    assert_eq!(
      got.map(f64::to_bits),
      pinned.map(f64::to_bits),
      "seed {seed}: {got:?}"
    );
    assert_eq!(
      fit.vol_path.as_slice().unwrap()[1..4]
        .iter()
        .map(|v| v.to_bits())
        .collect::<Vec<_>>(),
      variance.map(f64::to_bits).to_vec(),
      "seed {seed}"
    );
    assert_eq!((fit.iterations, fit.converged), (12, false), "seed {seed}");
  }
}

/// Under the batch's clamp a NaN or an overflow is `Numerical` at its stage, not floored to `EPS`.
#[test]
fn the_batch_clamp_answers_a_non_finite_step_with_a_numerical_error() {
  let (prices, _) = simulate_heston_prices(64, 1.0 / 252.0, 61);
  for (parameters, stage) in [
    (
      HestonNmleCekfParams {
        rho: f64::NAN,
        ..HestonNmleCekfParams::default()
      },
      "innovation covariance",
    ),
    (
      HestonNmleCekfParams {
        kappa: 1e300,
        ..HestonNmleCekfParams::default()
      },
      "predicted error covariance bound",
    ),
  ] {
    assert_eq!(
      filter_heston_cekf_batch(
        prices.view(),
        parameters,
        &HestonCekfFilterConfig::default()
      )
      .unwrap_err(),
      StatsError::Numerical { stage }
    );
  }
}

#[test]
fn the_batch_configuration_panics_in_the_predicate_form() {
  let (prices, _) = simulate_heston_prices(16, 1.0 / 252.0, 61);
  let base = HestonNmleCekfConfig::default();
  for (config, message) in [
    (
      HestonNmleCekfConfig {
        r: f64::NAN,
        ..base.clone()
      },
      "r must satisfy `r.is_finite()`, got r = NaN",
    ),
    (
      HestonNmleCekfConfig {
        delta: 0.0,
        ..base.clone()
      },
      "delta must satisfy `delta > 0.0`, got delta = 0.0",
    ),
    (
      HestonNmleCekfConfig {
        max_iters: 0,
        ..base.clone()
      },
      "max_iters must satisfy `max_iters > 0`, got max_iters = 0",
    ),
    (
      HestonNmleCekfConfig {
        tol: f64::INFINITY,
        ..base.clone()
      },
      "tol must satisfy `tol.is_finite()`, got tol = inf",
    ),
    (
      HestonNmleCekfConfig {
        param_damping: f64::NAN,
        ..base.clone()
      },
      "param_damping must satisfy `param_damping.is_finite()`, got param_damping = NaN",
    ),
    (
      HestonNmleCekfConfig {
        initial_v0: -0.04,
        ..base.clone()
      },
      "initial_v0 must satisfy `initial_v0 > 0.0`, got initial_v0 = -0.04",
    ),
    (
      HestonNmleCekfConfig {
        initial_p0: 0.0,
        ..base.clone()
      },
      "initial_p0 must satisfy `initial_p0 > 0.0`, got initial_p0 = 0.0",
    ),
    (
      HestonNmleCekfConfig {
        initial_params: HestonNmleCekfParams {
          kappa: f64::NAN,
          ..HestonNmleCekfParams::default()
        },
        ..base.clone()
      },
      "initial_params.kappa must satisfy `initial_params.kappa.is_finite()`, got initial_params.kappa = NaN",
    ),
    (
      HestonNmleCekfConfig {
        q22: f64::INFINITY,
        ..base.clone()
      },
      "q22 must satisfy `q22.is_finite()`, got q22 = inf",
    ),
    (
      HestonNmleCekfConfig {
        q12: 2.0,
        ..base.clone()
      },
      "q11 * q22 - q12 * q12 must satisfy `>= -EPS`, got q11 * q22 - q12 * q12 = -3.0",
    ),
  ] {
    let prices = prices.clone();
    let text = panic_text(move || {
      let _ = fit_heston_nmle_cekf(prices.view(), config);
    });
    assert_eq!(text, message);
  }
}
