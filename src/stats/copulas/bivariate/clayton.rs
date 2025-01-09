use ndarray::{Array1, Array2};
use std::f64;

use super::{Bivariate, CopulaType};

/// `BaseBivariate` struct – Közös mezők a bivariate copulákhoz
#[derive(Debug, Clone)]
pub struct Clayton {
  pub r#type: CopulaType,
  pub theta: Option<f64>,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Bivariate for Clayton {
  fn r#type(&self) -> CopulaType {
    self.r#type
  }

  fn tau(&self) -> Option<f64> {
    self.tau
  }

  fn theta(&self) -> Option<f64> {
    self.theta
  }

  fn theta_bounds(&self) -> (f64, f64) {
    self.theta_bounds
  }

  fn invalid_thetas(&self) -> Vec<f64> {
    self.invalid_thetas.clone()
  }

  fn set_theta(&mut self, theta: f64) {
    self.theta = Some(theta);
  }

  fn generator(&self, t: &Array1<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    todo!()
  }

  fn probability_density(
    &self,
    X: &Array2<f64>,
  ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    todo!()
  }

  fn cumulative_distribution(
    &self,
    X: &Array2<f64>,
  ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    todo!()
  }

  fn percent_point(
    &self,
    y: &Array1<f64>,
    V: &Array1<f64>,
  ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    todo!()
  }

  fn partial_derivative(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    todo!()
  }

  fn compute_theta(&self) -> f64 {
    if self.tau.is_some() && self.tau.unwrap() == 1.0 {
      return f64::INFINITY;
    }

    let tau = self.tau.unwrap();

    2.0 * tau / (1.0 - tau)
  }
}
