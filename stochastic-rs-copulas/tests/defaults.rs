//! `Frank` and `Gumbel` implement `Default`, so struct-update syntax over it gives a finite-sampling copula with one parameter changed.
//! Neither holds RNG state, so a clone replays the same draws as the original under the same explicit seed.

use stochastic_rs_copulas::BivariateExt;
use stochastic_rs_copulas::bivariate::frank::Frank;
use stochastic_rs_copulas::bivariate::gumbel::Gumbel;
use stochastic_rs_core::simd_rng::Deterministic;

#[test]
fn defaults_sample_finite() {
  let frank = Frank {
    theta: Some(6.0),
    ..Frank::default()
  };
  let out = frank.sample(16, &Deterministic::new(42)).unwrap();
  assert_eq!(out.nrows(), 16);
  assert!(out.iter().all(|x| x.is_finite()));

  let gumbel = Gumbel {
    theta: Some(2.0),
    ..Gumbel::default()
  };
  let out = gumbel.sample(16, &Deterministic::new(42)).unwrap();
  assert_eq!(out.nrows(), 16);
  assert!(out.iter().all(|x| x.is_finite()));
}

/// `Frank` and `Gumbel` hold no seed state, so a clone draws exactly what the original draws under the same `Deterministic`
/// seed, unlike a `ProcessExt` clone.
#[test]
fn clone_is_a_plain_value_copy() {
  let frank = Frank {
    theta: Some(6.0),
    ..Frank::default()
  };
  let cloned = frank.clone();
  assert_eq!(
    frank.sample(16, &Deterministic::new(42)).unwrap(),
    cloned.sample(16, &Deterministic::new(42)).unwrap()
  );

  let gumbel = Gumbel {
    theta: Some(2.0),
    ..Gumbel::default()
  };
  let cloned = gumbel.clone();
  assert_eq!(
    gumbel.sample(16, &Deterministic::new(42)).unwrap(),
    cloned.sample(16, &Deterministic::new(42)).unwrap()
  );
}
