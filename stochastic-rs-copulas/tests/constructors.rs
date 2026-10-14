//! `new` panics outside the family's domain, `try_new` and `from_tau` return the error, and the parameters are
//! private: a constructed copula always samples.

use ndarray::array;
use stochastic_rs_copulas::CopulaError;
use stochastic_rs_copulas::bivariate::amh::Amh;
use stochastic_rs_copulas::bivariate::bb1::Bb1;
use stochastic_rs_copulas::bivariate::bb7::Bb7;
use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::bivariate::fgm::Fgm;
use stochastic_rs_copulas::bivariate::frank::Frank;
use stochastic_rs_copulas::bivariate::galambos::Galambos;
use stochastic_rs_copulas::bivariate::gaussian::GaussianCopula;
use stochastic_rs_copulas::bivariate::gumbel::Gumbel;
use stochastic_rs_copulas::bivariate::husler_reiss::HuslerReiss;
use stochastic_rs_copulas::bivariate::independence::Independence;
use stochastic_rs_copulas::bivariate::joe::Joe;
use stochastic_rs_copulas::bivariate::marshall_olkin::MarshallOlkin;
use stochastic_rs_copulas::bivariate::plackett::Plackett;
use stochastic_rs_copulas::bivariate::t_copula::TCopula;
use stochastic_rs_copulas::multivariate::gaussian::GaussianMultivariate;
use stochastic_rs_copulas::multivariate::t::TMultivariate;
use stochastic_rs_copulas::simd_rng::Deterministic;
use stochastic_rs_copulas::traits::BivariateExt;
use stochastic_rs_copulas::traits::MultivariateExt;

fn is_parameter_error<T>(result: Result<T, CopulaError>, name: &str) -> bool {
  matches!(result, Err(CopulaError::InvalidParameter { name: n, .. }) if n == name)
}

#[test]
fn try_new_and_from_tau_report_the_domain() {
  assert!(is_parameter_error(Clayton::try_new(-1.0), "theta"));
  assert!(is_parameter_error(Clayton::from_tau(-0.2), "tau"));
  assert!(is_parameter_error(Gumbel::try_new(0.5), "theta"));
  assert!(is_parameter_error(Bb1::try_new(1.0, 0.5), "delta"));
  assert!(is_parameter_error(Bb7::try_new(0.5, 1.0), "theta"));
  assert!(is_parameter_error(TCopula::try_new(1.0, 4.0), "theta"));
  assert!(is_parameter_error(TCopula::try_new(0.5, 0.0), "nu"));
  assert!(is_parameter_error(
    MarshallOlkin::try_new(0.0, 0.5),
    "alpha"
  ));
  assert!(is_parameter_error(MarshallOlkin::try_new(0.5, 1.5), "beta"));
  assert!(is_parameter_error(GaussianCopula::try_new(1.0), "theta"));
  assert!(is_parameter_error(Fgm::from_tau(0.5), "tau"));
  assert_eq!(Clayton::new(2.0).theta(), Some(2.0));
  assert_eq!(Clayton::from_tau(0.5).unwrap().theta(), Some(2.0));
  assert_eq!(TCopula::from_tau(0.5, 6.0).unwrap().nu(), 6.0);
  assert_eq!(Bb1::from_tau(0.6, 1.6).unwrap().delta(), 1.6);
  let mo = MarshallOlkin::from_tau(0.25).unwrap();
  assert_eq!(
    (mo.alpha(), mo.beta(), mo.theta()),
    (Some(0.4), Some(0.4), Some(0.4))
  );
  assert_eq!(MarshallOlkin::new(0.3, 0.6).theta(), None);
  assert_eq!(Independence::new().theta(), Some(0.0));
}

/// A NaN is outside every domain and an infinite parameter outside every finite one, so neither constructs.
#[test]
fn a_nan_or_infinite_parameter_does_not_construct() {
  for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
    assert!(is_parameter_error(Amh::try_new(bad), "theta"), "{bad}");
    assert!(is_parameter_error(Clayton::try_new(bad), "theta"), "{bad}");
    assert!(is_parameter_error(Fgm::try_new(bad), "theta"), "{bad}");
    assert!(is_parameter_error(Frank::try_new(bad), "theta"), "{bad}");
    assert!(is_parameter_error(Galambos::try_new(bad), "theta"), "{bad}");
    assert!(
      is_parameter_error(GaussianCopula::try_new(bad), "theta"),
      "{bad}"
    );
    assert!(is_parameter_error(Gumbel::try_new(bad), "theta"), "{bad}");
    assert!(
      is_parameter_error(HuslerReiss::try_new(bad), "theta"),
      "{bad}"
    );
    assert!(is_parameter_error(Joe::try_new(bad), "theta"), "{bad}");
    assert!(is_parameter_error(Plackett::try_new(bad), "theta"), "{bad}");
    assert!(is_parameter_error(Bb1::try_new(bad, 1.6), "theta"), "{bad}");
    assert!(is_parameter_error(Bb1::try_new(0.8, bad), "delta"), "{bad}");
    assert!(is_parameter_error(Bb7::try_new(bad, 0.9), "theta"), "{bad}");
    assert!(is_parameter_error(Bb7::try_new(1.7, bad), "delta"), "{bad}");
    assert!(
      is_parameter_error(TCopula::try_new(bad, 4.0), "theta"),
      "{bad}"
    );
    assert!(
      is_parameter_error(TCopula::try_new(0.5, bad), "nu"),
      "{bad}"
    );
    assert!(
      is_parameter_error(MarshallOlkin::try_new(bad, 0.5), "alpha"),
      "{bad}"
    );
    assert!(
      is_parameter_error(MarshallOlkin::try_new(0.5, bad), "beta"),
      "{bad}"
    );
  }
}

#[test]
#[should_panic(expected = "theta must satisfy `0.0 <= theta < inf`, got theta = -1.0")]
fn new_panics_outside_the_domain() {
  let _ = Clayton::new(-1.0);
}

#[test]
#[should_panic(expected = "theta must satisfy `-inf < theta < inf`, got theta = inf")]
fn set_theta_panics_outside_the_domain() {
  Frank::default().set_theta(f64::INFINITY);
}

#[test]
#[should_panic(expected = "delta must satisfy `1.0 <= delta < inf`, got delta = 0.5")]
fn set_delta_panics_outside_the_domain() {
  Bb1::default().set_delta(0.5);
}

#[test]
#[should_panic(expected = "nu must satisfy `0.0 < nu < inf`, got nu = -5.0")]
fn set_nu_panics_outside_the_domain() {
  TCopula::default().set_nu(-5.0);
}

#[test]
fn every_constructed_copula_samples_and_every_default_is_unfitted() {
  let seed = Deterministic::new(1);
  let samples = [
    Amh::new(0.5).sample(16, &seed),
    Bb1::new(0.8, 1.6).sample(16, &seed),
    Bb7::new(1.7, 0.9).sample(16, &seed),
    Clayton::new(2.0).sample(16, &seed),
    Fgm::new(0.5).sample(16, &seed),
    Frank::new(4.0).sample(16, &seed),
    Galambos::new(1.5).sample(16, &seed),
    GaussianCopula::new(0.5).sample(16, &seed),
    Gumbel::new(2.0).sample(16, &seed),
    HuslerReiss::new(1.5).sample(16, &seed),
    Independence::new().sample(16, &seed),
    Joe::new(2.0).sample(16, &seed),
    MarshallOlkin::new(0.3, 0.6).sample(16, &seed),
    Plackett::new(3.0).sample(16, &seed),
    TCopula::new(0.5, 4.0).sample(16, &seed),
  ];
  assert!(
    samples
      .iter()
      .all(|s| s.as_ref().is_ok_and(|m| m.dim() == (16, 2)))
  );
  assert_eq!(
    Clayton::default().sample(4, &seed),
    Err(CopulaError::NotFitted)
  );
  assert_eq!(Bb7::default().sample(4, &seed), Err(CopulaError::NotFitted));
  assert_eq!(
    MarshallOlkin::default().sample(4, &seed),
    Err(CopulaError::NotFitted)
  );
}

#[test]
fn elliptical_constructors_validate_the_correlation_and_nu() {
  let corr = array![[1.0, 0.5], [0.5, 1.0]];
  assert!(matches!(
    GaussianMultivariate::try_new(array![[1.0, 0.5], [0.4, 1.0]]),
    Err(CopulaError::InvalidStructure(_))
  ));
  assert!(matches!(
    TMultivariate::try_new(array![[1.0]], 5.0),
    Err(CopulaError::InvalidStructure(_))
  ));
  for bad in [f64::INFINITY, f64::NAN, 0.0, -1.0, 5e-324] {
    assert!(
      is_parameter_error(TMultivariate::try_new(corr.clone(), bad), "nu"),
      "{bad}"
    );
  }
  assert!(
    GaussianMultivariate::new(corr.clone())
      .sample(8, &Deterministic::new(1))
      .is_ok()
  );
  assert!(
    TMultivariate::new(corr.clone(), 5.0)
      .sample(8, &Deterministic::new(1))
      .is_ok()
  );
  assert_eq!(
    GaussianMultivariate::try_new(corr.clone())
      .unwrap()
      .correlation(),
    Some(&corr)
  );
  assert_eq!(GaussianMultivariate::default().correlation(), None);
  assert_eq!(
    GaussianMultivariate::default().sample(8, &Deterministic::new(1)),
    Err(CopulaError::NotFitted)
  );
  assert_eq!(
    TMultivariate::default().sample(8, &Deterministic::new(1)),
    Err(CopulaError::NotFitted)
  );
}

#[test]
#[should_panic(
  expected = "invalid structure: corr must satisfy `|corr[[i, i]] - 1.0| <= 1e-12` (unit diagonal), got corr[[0, 0]] = 1.1"
)]
fn an_elliptical_new_panics_on_a_bad_matrix() {
  let _ = GaussianMultivariate::new(array![[1.1, 0.5], [0.5, 1.0]]);
}

#[test]
#[should_panic(expected = "nu must satisfy `0.0 < nu < inf`, got nu = -5.0")]
fn the_multivariate_t_new_panics_outside_the_domain() {
  let _ = TMultivariate::new(array![[1.0, 0.5], [0.5, 1.0]], -5.0);
}

#[test]
#[should_panic(expected = "nu must satisfy `0.0 < nu < inf`, got nu = -5.0")]
fn the_multivariate_set_nu_panics_outside_the_domain() {
  TMultivariate::default().set_nu(-5.0);
}
