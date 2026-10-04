//! `CopulaError` is one `Send + Sync` enum: it crosses threads, `?` converts into `anyhow`, and a message names its parameter.

use ndarray::Array2;
use ndarray::array;
use stochastic_rs_copulas::CopulaError;
use stochastic_rs_copulas::bivariate::bb1::Bb1;
use stochastic_rs_copulas::bivariate::bb7::Bb7;
use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::bivariate::gaussian::GaussianCopula;
use stochastic_rs_copulas::multivariate::fit::PairFamily;
use stochastic_rs_copulas::multivariate::fit::SelectionCriterion;
use stochastic_rs_copulas::multivariate::fit::VineStructure;
use stochastic_rs_copulas::multivariate::fit::fit_vine;
use stochastic_rs_copulas::multivariate::gaussian::GaussianMultivariate;
use stochastic_rs_copulas::traits::BivariateExt;
use stochastic_rs_copulas::traits::MultivariateExt;

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
  assert_eq!(e.to_string(), "nu must satisfy `nu > 0`, got nu = 0.0");
}

#[test]
fn a_bivariate_fit_needs_two_observations() {
  let one = array![[0.5_f64, 0.5]];
  let short = CopulaError::InsufficientData { needed: 2, got: 1 };
  assert_eq!(Clayton::new().fit(&one).unwrap_err(), short);
  assert_eq!(Bb1::default().fit(&one).unwrap_err(), short);
  assert_eq!(Bb7::default().fit(&one).unwrap_err(), short);
}

#[test]
fn a_multivariate_fit_tells_too_few_dimensions_from_too_few_observations() {
  let mut gaussian = GaussianMultivariate::new();
  let one_column = gaussian.fit(Array2::from_elem((100, 1), 0.5)).unwrap_err();
  assert!(
    matches!(one_column, CopulaError::InvalidStructure(_)),
    "{one_column:?}"
  );
  assert_eq!(
    gaussian.fit(Array2::from_elem((1, 3), 0.5)).unwrap_err(),
    CopulaError::InsufficientData { needed: 2, got: 1 }
  );
  let vine = fit_vine(
    &Array2::from_elem((100, 1), 0.5),
    VineStructure::DVine,
    &PairFamily::ALL,
    SelectionCriterion::Aic,
  );
  assert!(matches!(vine, Err(CopulaError::InvalidStructure(_))));
}

#[test]
fn an_out_of_domain_theta_names_its_bounds_and_only_a_real_exclusion_set() {
  let clayton = Clayton {
    theta: Some(-1.0),
    ..Clayton::new()
  };
  assert_eq!(
    clayton.check_theta().unwrap_err().to_string(),
    "theta must satisfy `0.0 <= theta <= inf`, got theta = -1.0"
  );
  let gaussian = GaussianCopula {
    theta: Some(1.0),
    ..GaussianCopula::new()
  };
  assert_eq!(
    gaussian.check_theta().unwrap_err(),
    CopulaError::InvalidParameter {
      name: "theta",
      value: 1.0,
      constraint: "-1.0 <= theta <= 1.0, theta not in [-1.0, 1.0]".into(),
    }
  );
}
