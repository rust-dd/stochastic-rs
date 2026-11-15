//! Data-dependent failures and unchecked struct-literal fields are `Err`; validated configuration panics.

use ndarray::Array1;
use stochastic_rs_stats::StatsError;
use stochastic_rs_stats::fractal_dim::FractalDimEstimator;
use stochastic_rs_stats::fractal_dim::Higuchi;
use stochastic_rs_stats::heston_nml_cekf::HestonCekfFilterConfig;
use stochastic_rs_stats::heston_nml_cekf::HestonCekfState;
use stochastic_rs_stats::heston_nml_cekf::HestonNmleCekfParams;
use stochastic_rs_stats::heston_nml_cekf::heston_cekf_step;
use stochastic_rs_stats::hurst::HurstEstimator;
use stochastic_rs_stats::hurst::rs::RescaledRange;
use stochastic_rs_stats::log_realized_variance_qml::log_chi_square_moments;

fn assert_send_sync<T: Send + Sync + std::error::Error + 'static>() {}

#[test]
fn the_error_is_send_sync_and_displays_the_predicate() {
  assert_send_sync::<StatsError>();
  assert_eq!(
    StatsError::InsufficientData { needed: 32, got: 5 }.to_string(),
    "needs at least 32 observations, got 5"
  );
  assert_eq!(
    StatsError::InvalidParameter {
      name: "kmax",
      value: 1.0,
      constraint: "kmax >= 2"
    }
    .to_string(),
    "kmax must satisfy `kmax >= 2`, got kmax = 1.0"
  );
  assert_eq!(
    StatsError::Singular("S00").to_string(),
    "S00 has no finite solution"
  );
  assert_eq!(
    StatsError::Numerical { stage: "filter" }.to_string(),
    "filter produced no usable value"
  );
}

#[test]
fn a_short_path_is_insufficient_data() {
  let short = Array1::<f64>::linspace(0.0, 1.0, 5);
  assert_eq!(
    RescaledRange::default().estimate(short.view()).unwrap_err(),
    StatsError::InsufficientData { needed: 32, got: 5 }
  );
}

#[test]
fn a_bad_estimator_configuration_is_invalid_parameter() {
  let x = Array1::<f64>::linspace(0.0, 1.0, 64);
  let err = FractalDimEstimator::<f64>::estimate(&Higuchi { kmax: 1 }, x.view()).unwrap_err();
  assert!(
    matches!(err, StatsError::InvalidParameter { name: "kmax", .. }),
    "{err}"
  );
}

#[test]
fn a_non_finite_log_return_is_an_invalid_observation() {
  let previous = HestonCekfState {
    variance: 0.04,
    error_covariance_bound: 0.0004,
  };
  let err = heston_cekf_step(
    previous,
    f64::NAN,
    HestonNmleCekfParams::default(),
    &HestonCekfFilterConfig::default(),
  )
  .unwrap_err();
  assert!(
    matches!(
      err,
      StatsError::InvalidObservation {
        constraint: "finite",
        ..
      }
    ),
    "{err}"
  );
}

#[test]
#[should_panic(expected = "delta must satisfy `delta > 0.0`")]
fn a_bad_filter_configuration_panics() {
  let cfg = HestonCekfFilterConfig {
    delta: -1.0,
    ..HestonCekfFilterConfig::default()
  };
  let previous = HestonCekfState {
    variance: 0.04,
    error_covariance_bound: 0.0004,
  };
  let _ = heston_cekf_step(previous, 0.001, HestonNmleCekfParams::default(), &cfg);
}

#[test]
#[should_panic(expected = "block_degrees_of_freedom must satisfy `block_degrees_of_freedom > 0`")]
fn a_zero_block_size_panics() {
  let _ = log_chi_square_moments(0);
}

#[test]
fn the_display_of_a_data_error_names_the_index() {
  let err = StatsError::InvalidObservation {
    index: 3,
    value: -0.5,
    constraint: "finite and > 0",
  };
  assert_eq!(
    err.to_string(),
    "observation 3 = -0.5 must satisfy `finite and > 0`"
  );
}
