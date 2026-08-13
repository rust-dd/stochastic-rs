// docs: copulas#gaussian-copula-multivariate-sampling
//! Backs the Gaussian multivariate copula example on the copulas catalog
//! page. Needs `openblas` for the Cholesky factorisation, so the whole
//! file is gated on that feature.

#![cfg(feature = "openblas")]

use ndarray::array;
use stochastic_rs::copulas::multivariate::gaussian::GaussianMultivariate;
use stochastic_rs::traits::MultivariateExt;

#[test]
fn gaussian_multivariate_sample() {
  let corr = array![[1.0, 0.7, 0.3], [0.7, 1.0, 0.5], [0.3, 0.5, 1.0]];
  let cop = GaussianMultivariate::new_with_corr(corr).unwrap();
  let samples = cop.sample(10_000).unwrap(); // Array2<f64>, shape (10_000, 3)
  assert_eq!(samples.dim(), (10_000, 3));
}
