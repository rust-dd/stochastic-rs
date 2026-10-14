//! Every input is an `ArrayView`: a strided column pair of a wider matrix reads exactly like its dense copy.

use ndarray::Array2;
use ndarray::s;
use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::gof::pseudo_observations;
use stochastic_rs_copulas::multivariate::gaussian::GaussianMultivariate;
use stochastic_rs_copulas::traits::BivariateExt;
use stochastic_rs_copulas::traits::MultivariateExt;
use stochastic_rs_core::simd_rng::Deterministic;

#[test]
fn strided_views_read_like_their_dense_copy() {
  let truth = Clayton::new(2.0);
  let mut wide = Array2::<f64>::zeros((400, 4));
  wide
    .slice_mut(s![.., 1..3])
    .assign(&truth.sample(400, &Deterministic::new(3)).unwrap());
  let pair = wide.slice(s![.., 1..3]);
  assert!(!pair.is_standard_layout());
  let dense = pair.to_owned();

  let mut fitted = Clayton::default();
  fitted.fit(pair).unwrap();
  let mut fitted_dense = Clayton::default();
  fitted_dense.fit(dense.view()).unwrap();
  assert!((fitted.theta().unwrap() - 2.0).abs() < 0.6);
  assert_eq!(
    fitted.theta().unwrap().to_bits(),
    fitted_dense.theta().unwrap().to_bits()
  );
  assert_eq!(fitted.pdf(pair).unwrap(), fitted.pdf(dense.view()).unwrap());
  assert_eq!(
    fitted
      .percent_point(pair.column(0), pair.column(1))
      .unwrap(),
    fitted
      .percent_point(dense.column(0), dense.column(1))
      .unwrap()
  );

  let observations = pseudo_observations(wide.view());
  let strided = observations.slice(s![.., 1..3]);
  assert!(!strided.is_standard_layout());
  let mut gaussian = GaussianMultivariate::default();
  gaussian.fit(strided).unwrap();
  let mut gaussian_dense = GaussianMultivariate::default();
  gaussian_dense.fit(strided.to_owned().view()).unwrap();
  assert_eq!(
    gaussian.correlation().unwrap(),
    gaussian_dense.correlation().unwrap()
  );
}
