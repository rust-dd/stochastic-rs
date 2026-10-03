//! The umbrella re-exports the crates whose types callers build and pass across its API.

use stochastic_rs::chrono::NaiveDate;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::ndarray::Array1;
use stochastic_rs::num_complex::Complex64;
use stochastic_rs::num_traits::Zero;
use stochastic_rs::prelude::*;
use stochastic_rs::quant::calendar::DayCountConvention;
use stochastic_rs::rand::distr::Distribution;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::process::bm::Bm;

fn path_len(path: Array1<f64>) -> usize {
  path.len()
}

fn is_zero(z: Complex64) -> bool {
  z.is_zero()
}

fn is_distribution<D: Distribution<f64>>(_: &D) {}

fn _real_ext_is_the_reexported_float<T: RealExt>() {
  fn float<U: stochastic_rs::num_traits::Float>() {}
  float::<T>();
}

#[test]
fn reexported_ndarray_is_the_array_type_of_the_api() {
  let path = Bm::<f64>::new(16, Some(1.0), Unseeded).sample();
  assert_eq!(path_len(path), 16);
}

#[test]
fn reexported_num_complex_is_the_complex_type_of_the_api() {
  let normal = SimdNormal::<f64>::new(0.0, 1.0);
  assert!(!is_zero(normal.characteristic_function(1.0)));
}

#[test]
fn reexported_rand_is_the_distribution_trait_of_the_api() {
  is_distribution(&SimdNormal::<f64>::new(0.0, 1.0));
}

#[test]
fn reexported_chrono_is_the_date_type_of_the_api() {
  let start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
  let end = NaiveDate::from_ymd_opt(2027, 1, 1).unwrap();
  let tau = DayCountConvention::Actual365Fixed.year_fraction::<f64>(start, end);
  assert!((tau - 1.0).abs() < 1e-12);
}
