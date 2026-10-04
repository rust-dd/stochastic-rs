//! `CopulaError` is one `Send + Sync` enum: it crosses threads, `?` converts into `anyhow`, and a message names its parameter.

use ndarray::Array2;
use stochastic_rs_copulas::CopulaError;
use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::traits::BivariateExt;

fn assert_send_sync<T: Send + Sync + std::error::Error + 'static>() {}

#[test]
fn the_error_crosses_threads_and_converts_into_anyhow() {
  assert_send_sync::<CopulaError>();
  let unset = Clayton::new().sample_with_seed(8, 7);
  assert_eq!(unset.unwrap_err(), CopulaError::NotFitted);
  let into_anyhow: anyhow::Result<Array2<f64>> = Clayton::new()
    .sample_with_seed(8, 7)
    .map_err(anyhow::Error::from);
  assert!(into_anyhow.is_err());
}

#[test]
fn an_invalid_parameter_names_itself_in_the_crate_form() {
  let e = CopulaError::InvalidParameter {
    name: "nu",
    value: 0.0,
    constraint: "nu > 0".into(),
  };
  assert_eq!(e.to_string(), "nu must satisfy `nu > 0`, got nu = 0");
}
