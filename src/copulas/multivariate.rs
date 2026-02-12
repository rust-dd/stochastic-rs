use std::error::Error;

use ndarray::Array1;
use ndarray::Array2;

pub mod gaussian;
pub mod tree;
pub mod vine;

pub enum CopulaType {
  Gaussian,
  Tree,
  Vine,
}

pub trait MultivariateExt {
  fn r#type(&self) -> CopulaType;

  fn sample(&self, n: usize) -> Result<Array2<f64>, Box<dyn Error>>;

  fn fit(&mut self, X: Array2<f64>) -> Result<(), Box<dyn Error>>;

  fn check_fit(&self, X: &Array2<f64>) -> Result<(), Box<dyn Error>>;

  fn pdf(&self, X: Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>>;

  fn log_pdf(&self, X: Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    Ok(self.pdf(X)?.ln())
  }

  fn cdf(&self, X: Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>>;
}
