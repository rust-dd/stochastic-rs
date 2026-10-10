//! `MultivariateExt` — feature-gated multivariate copula trait.

use ndarray::Array1;

use crate::error::CopulaError;
use crate::multivariate::CopulaType as MultivariateCopulaType;

pub trait MultivariateExt {
  fn r#type(&self) -> MultivariateCopulaType;

  fn sample(&self, n: usize) -> Result<ndarray::Array2<f64>, CopulaError>;

  /// Deterministic sampler. Returns the same matrix for a fixed `seed`,
  /// mirroring [`crate::traits::BivariateExt::sample_with_seed`].
  fn sample_with_seed(&self, n: usize, seed: u64) -> Result<ndarray::Array2<f64>, CopulaError>;

  fn fit(&mut self, X: ndarray::Array2<f64>) -> Result<(), CopulaError>;

  fn check_fit(&self, X: &ndarray::Array2<f64>) -> Result<(), CopulaError>;

  fn pdf(&self, X: &ndarray::Array2<f64>) -> Result<Array1<f64>, CopulaError>;

  fn log_pdf(&self, X: &ndarray::Array2<f64>) -> Result<Array1<f64>, CopulaError> {
    Ok(self.pdf(X)?.ln())
  }

  fn cdf(&self, X: &ndarray::Array2<f64>) -> Result<Array1<f64>, CopulaError>;
}
