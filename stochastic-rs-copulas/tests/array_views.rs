//! Every input is an `ArrayView`: a strided column pair of a wider matrix fits, evaluates and inverts without a copy.

use ndarray::Array2;
use ndarray::s;
use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::gof::pseudo_observations;
use stochastic_rs_copulas::multivariate::gaussian::GaussianMultivariate;
use stochastic_rs_copulas::traits::BivariateExt;
use stochastic_rs_copulas::traits::MultivariateExt;

#[test]
fn strided_views_are_accepted_without_a_copy() {
  let truth = Clayton {
    theta: Some(2.0),
    ..Clayton::new()
  };
  let mut wide = Array2::<f64>::zeros((400, 4));
  wide
    .slice_mut(s![.., 1..3])
    .assign(&truth.sample_with_seed(400, 3).unwrap());
  let pair = wide.slice(s![.., 1..3]);
  assert!(!pair.is_standard_layout());
  let mut fitted = Clayton::new();
  fitted.fit(pair).unwrap();
  assert!((fitted.theta().unwrap() - 2.0).abs() < 0.6);
  assert_eq!(fitted.pdf(pair).unwrap().len(), 400);
  assert_eq!(
    fitted
      .percent_point(pair.column(0), pair.column(1))
      .unwrap()
      .len(),
    400
  );
  let mut gaussian = GaussianMultivariate::new();
  gaussian
    .fit(pseudo_observations(wide.view()).view())
    .unwrap();
  assert_eq!(gaussian.correlation().unwrap().nrows(), 4);
}
