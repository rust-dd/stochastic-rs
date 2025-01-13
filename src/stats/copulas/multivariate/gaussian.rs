use std::error::Error;

use ndarray::{Array1, Array2};

use super::{CopulaType, Multivariate};

#[derive(Debug)]
pub struct GaussianMultivariate;

impl GaussianMultivariate {
  pub fn new() -> Self {
    Self
  }

  fn transform_to_normal(&self) {}

  fn fit_columns(&self) {}

  fn get_distribution_for_column(&self) {}

  fn fit_column(&self) {}

  fn fit_with_fallback_distribution(&self) {}

  fn get_correlation_matrix(&self) {}

  fn get_conditional_distribution(&self) {}

  fn get_normal_samples(&self) {}
}

impl Multivariate for GaussianMultivariate {
  fn r#type(&self) -> CopulaType {
    CopulaType::Gaussian
  }

  fn sample(&self, n: usize) -> Result<Array2<f64>, Box<dyn Error>> {
    todo!()
  }

  fn fit(&mut self, X: Array2<f64>) -> Result<(), Box<dyn Error>> {
    todo!()
  }

  fn check_fit(&self, X: &Array2<f64>) -> Result<(), Box<dyn Error>> {
    todo!()
  }

  fn pdf(&self, X: Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    todo!()
  }

  fn log_pdf(&self, X: Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    todo!()
  }

  fn cdf(&self, X: Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    todo!()
  }
}
