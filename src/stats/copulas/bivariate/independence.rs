use std::error::Error;

use ndarray::{Array1, Array2, Axis};

use super::{Bivariate, CopulaType};

#[derive(Debug, Clone)]
pub struct Independence {
  pub r#type: CopulaType,
  pub theta: Option<f64>,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Independence {
  pub fn new() -> Self {
    Self {
      r#type: CopulaType::Independence,
      theta: None,
      tau: None,
      theta_bounds: (0.0, 0.0),
      invalid_thetas: vec![],
    }
  }
}

impl Bivariate for Independence {
  fn r#type(&self) -> CopulaType {
    self.r#type
  }

  fn tau(&self) -> Option<f64> {
    self.tau
  }

  fn set_tau(&mut self, tau: f64) {
    self.tau = Some(tau);
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

  fn fit(&mut self, _X: &Array2<f64>) -> Result<(), Box<dyn Error>> {
    Ok(())
  }

  fn generator(&self, t: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    Ok(t.ln())
  }

  fn pdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    let in_range = X.map_axis(Axis(1), |row| {
      row.iter().all(|&val| val >= 0.0 && val <= 1.0)
    });

    let out = in_range.map(|&val| if val { 1.0 } else { 0.0 });

    Ok(out)
  }

  fn cdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    let U = X.column(0);
    let V = X.column(1);

    Ok(&U * &V)
  }

  fn partial_derivative(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    let V = X.column(1);
    Ok(V.to_owned())
  }

  fn percent_point(
    &self,
    y: &Array1<f64>,
    _V: &Array1<f64>,
  ) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;
    Ok(y.to_owned())
  }

  fn compute_theta(&self) -> f64 {
    panic!("There is no theta to calculate")
  }
}
