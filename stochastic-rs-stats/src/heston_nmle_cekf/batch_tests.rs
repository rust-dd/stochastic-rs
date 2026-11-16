use std::panic::UnwindSafe;
use std::panic::catch_unwind;

use super::HestonCekfConsistencyBounds;
use super::HestonCekfCorrection;
use super::HestonCekfFilterConfig;
use super::HestonNmleCekfConfig;
use super::HestonNmleCekfCorrection;
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

fn prior_bounds(state_transition: f64, kappa_theta: f64, sigma: f64) -> HestonNmleCekfCorrection {
  HestonNmleCekfCorrection::PriorBounds(HestonCekfConsistencyBounds {
    max_abs_state_transition: state_transition,
    max_abs_kappa_theta: kappa_theta,
    max_abs_sigma: sigma,
  })
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
  let own = HestonCekfConsistencyBounds {
    max_abs_state_transition: (1.0 - parameters.kappa * delta).abs(),
    max_abs_kappa_theta: parameters.kappa * parameters.theta,
    max_abs_sigma: parameters.sigma,
  };
  let prior = HestonCekfConsistencyBounds {
    max_abs_state_transition: 1.0,
    max_abs_kappa_theta: 0.1,
    max_abs_sigma: 0.5,
  };
  for (correction, bounds) in [
    (HestonNmleCekfCorrection::IterateBounds, own),
    (HestonNmleCekfCorrection::PriorBounds(prior), prior),
  ] {
    let batch = HestonNmleCekfConfig {
      delta,
      initial_params: parameters,
      initial_p0: 0.0004,
      q11: 1.4,
      q12: 0.35,
      q22: 1.2,
      correction,
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
      correction: HestonCekfCorrection::Consistent { bounds },
      positive_state_policy: super::HestonCekfPositiveStatePolicy::Strict,
    };
    let corrected = filter_heston_cekf(prices.view(), parameters, &filter).unwrap();

    assert_eq!(batch_variance, corrected.variance_path, "{correction:?}");
    assert_eq!(
      batch_covariance, corrected.error_covariance_bound_path,
      "{correction:?}"
    );
  }
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

/// The default and the traditional correction reproduce their pinned fits bit for bit.
#[test]
fn each_correction_reproduces_its_pinned_fit_bit_for_bit() {
  assert_eq!(
    HestonNmleCekfConfig::default().correction,
    HestonNmleCekfCorrection::IterateBounds
  );
  for (observations, seed, correction, pinned, variance) in [
    (
      64,
      61,
      HestonNmleCekfCorrection::IterateBounds,
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
      HestonNmleCekfCorrection::IterateBounds,
      [
        0.04,
        196.68353068468784,
        0.01558564991592691,
        3.4982296798301187,
        -0.9998997343326441,
      ],
      [1e-12, 0.01217715390796855, 0.05835715408131442],
    ),
    (
      128,
      7,
      HestonNmleCekfCorrection::Traditional,
      [
        0.04,
        210.97775987240087,
        0.013317588502963738,
        3.3497468922263938,
        -0.9998997343326441,
      ],
      [1e-12, 0.0111513041338871, 0.05556334921863506],
    ),
  ] {
    let (prices, _) = simulate_heston_prices(observations, 1.0 / 252.0, seed);
    let config = HestonNmleCekfConfig {
      correction,
      ..HestonNmleCekfConfig::default()
    };
    let fit = fit_heston_nmle_cekf(prices.view(), config).unwrap();
    let params = fit.params;
    let got = [
      params.v0,
      params.kappa,
      params.theta,
      params.sigma,
      params.rho,
    ];
    let case = format!("seed {seed}, {correction:?}");
    assert_eq!(
      got.map(f64::to_bits),
      pinned.map(f64::to_bits),
      "{case}: {got:?}"
    );
    assert_eq!(
      fit.vol_path.as_slice().unwrap()[1..4]
        .iter()
        .map(|v| v.to_bits())
        .collect::<Vec<_>>(),
      variance.map(f64::to_bits).to_vec(),
      "{case}"
    );
    assert_eq!((fit.iterations, fit.converged), (12, false), "{case}");
  }
}

#[test]
fn prior_bounds_an_iterate_leaves_are_a_numerical_error() {
  let (prices, _) = simulate_heston_prices(64, 1.0 / 252.0, 61);
  let config = HestonNmleCekfConfig {
    correction: prior_bounds(1.0, 0.07, 0.5),
    ..HestonNmleCekfConfig::default()
  };
  assert_eq!(
    fit_heston_nmle_cekf(prices.view(), config).unwrap_err(),
    StatsError::Numerical {
      stage: "consistency bounds"
    }
  );
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
  let cases: [(fn(&mut HestonNmleCekfConfig), &str); 12] = [
    (
      |c| c.r = f64::NAN,
      "r must satisfy `r.is_finite()`, got r = NaN",
    ),
    (
      |c| c.delta = 0.0,
      "delta must satisfy `delta > 0.0`, got delta = 0.0",
    ),
    (
      |c| c.max_iters = 0,
      "max_iters must satisfy `max_iters > 0`, got max_iters = 0",
    ),
    (
      |c| c.tol = f64::INFINITY,
      "tol must satisfy `tol.is_finite()`, got tol = inf",
    ),
    (
      |c| c.param_damping = f64::NAN,
      "param_damping must satisfy `param_damping.is_finite()`, got param_damping = NaN",
    ),
    (
      |c| c.initial_v0 = -0.04,
      "initial_v0 must satisfy `initial_v0 > 0.0`, got initial_v0 = -0.04",
    ),
    (
      |c| c.initial_p0 = 0.0,
      "initial_p0 must satisfy `initial_p0 > 0.0`, got initial_p0 = 0.0",
    ),
    (
      |c| c.initial_params.kappa = f64::NAN,
      "initial_params.kappa must satisfy `initial_params.kappa.is_finite()`, got initial_params.kappa = NaN",
    ),
    (
      |c| c.q22 = f64::INFINITY,
      "q22 must satisfy `q22.is_finite()`, got q22 = inf",
    ),
    (
      |c| c.q12 = 2.0,
      "q11 * q22 - q12 * q12 must satisfy `q11 * q22 - q12 * q12 >= -1e-12`, got q11 * q22 - q12 * q12 = -3.0",
    ),
    (
      |c| c.correction = prior_bounds(1.0, 1.0, -1.0),
      "max_abs_sigma must satisfy `max_abs_sigma >= 0.0`, got max_abs_sigma = -1.0",
    ),
    (
      |c| c.correction = prior_bounds(1.0, 1e-6, 1e-6),
      "|kappa * theta| must satisfy `|kappa * theta| <= max_abs_kappa_theta + 1e-12`, got |kappa * theta| = 0.06, max_abs_kappa_theta = 1e-6",
    ),
  ];
  for (edit, message) in cases {
    let mut config = HestonNmleCekfConfig::default();
    edit(&mut config);
    let prices = prices.clone();
    let text = panic_text(move || {
      let _ = fit_heston_nmle_cekf(prices.view(), config);
    });
    assert_eq!(text, message);
  }
}
