use std::error::Error;

use ndarray::{Array1, Array2};
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

use crate::stats::copulas::univariate::gaussian::GaussianUnivariate;

use super::{CopulaType, Multivariate};

#[derive(Debug)]
pub struct GaussianMultivariate;

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
