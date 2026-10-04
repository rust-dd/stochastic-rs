//! `CopulaError` is one `Send + Sync` enum: it crosses threads, `?` converts into `anyhow`, and a message names its parameter.

use ndarray::Array2;
use ndarray::array;
use stochastic_rs_copulas::CopulaError;
use stochastic_rs_copulas::bivariate::bb1::Bb1;
use stochastic_rs_copulas::bivariate::bb7::Bb7;
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

#[test]
fn a_bivariate_fit_needs_two_observations() {
  let one = array![[0.5_f64, 0.5]];
  let short = CopulaError::InsufficientData { needed: 2, got: 1 };
  assert_eq!(Clayton::new().fit(&one).unwrap_err(), short);
  assert_eq!(Bb1::default().fit(&one).unwrap_err(), short);
  assert_eq!(Bb7::default().fit(&one).unwrap_err(), short);
}
