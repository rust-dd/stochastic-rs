//! `MultivariateExt` — feature-gated multivariate copula trait.

#![cfg(feature = "openblas")]

use std::error::Error;

use ndarray::Array1;

use crate::multivariate::CopulaType as MultivariateCopulaType;

pub trait MultivariateExt {
  fn r#type(&self) -> MultivariateCopulaType;

  fn sample(&self, n: usize) -> Result<ndarray::Array2<f64>, Box<dyn Error>>;

  fn fit(&mut self, X: ndarray::Array2<f64>) -> Result<(), Box<dyn Error>>;

  fn check_fit(&self, X: &ndarray::Array2<f64>) -> Result<(), Box<dyn Error>>;

  fn pdf(&self, X: ndarray::Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>>;

  fn log_pdf(&self, X: ndarray::Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    Ok(self.pdf(X)?.ln())
  }

  fn cdf(&self, X: ndarray::Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>>;
}
