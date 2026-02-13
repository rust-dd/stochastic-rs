use std::error::Error;
use std::f64;

use ndarray::Array1;
use ndarray::Array2;

use super::CopulaType;
use crate::traits::BivariateExt;

#[derive(Debug, Clone)]
pub struct Clayton {
  pub r#type: CopulaType,
  pub theta: Option<f64>,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Default for Clayton {
  fn default() -> Self {
    Self {
      r#type: CopulaType::Clayton,
      theta: None,
      tau: None,
      theta_bounds: (0.0, f64::INFINITY),
      invalid_thetas: vec![],
    }
  }
}

impl Clayton {
  pub fn new() -> Self {
    Self::default()
  }
}

impl BivariateExt for Clayton {
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

  fn generator(&self, t: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    let theta = self.theta.unwrap();
    Ok((1.0 / theta) * (t.powf(-theta) - 1.0))
  }

  fn pdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    let U = X.column(0);
    let V = X.column(1);

    let theta = self.theta.unwrap();
    let a = (theta + 1.0) * (&U * &V).powf(-theta - 1.0);
    let b = U.powf(-theta) + V.powf(-theta) - 1.0;
    let c = -(2.0 * theta + 1.0) / theta;
    Ok(a * b.powf(c))
  }

  fn cdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    let U = X.column(0);
    let V = X.column(1);

    let V_all_zeros = V.iter().all(|&v| v == 0.0);
    let U_all_zeros = U.iter().all(|&u| u == 0.0);

    if V_all_zeros || U_all_zeros {
      let shape = V.shape();
      return Ok(Array1::zeros(shape[0]));
    }

    let theta = self.theta.unwrap();
    let mut cdfs = Array1::<f64>::zeros(U.len());

    for i in 0..U.len() {
      let u = U[i];
      let v = V[i];

      if u > 0.0 && v > 0.0 {
        cdfs[i] = (u.powf(-theta) + v.powf(-theta) - 1.0).powf(-1.0 / theta);
      } else {
        cdfs[i] = 0.0;
      }
    }

    Ok(cdfs)
  }

  fn percent_point(&self, y: &Array1<f64>, V: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    let theta = self.theta.unwrap();

    if theta == 0.0 {
      return Ok(V.clone());
    }

    let a = y.powf(theta / (-1.0 - theta));
    let b = V.powf(theta);

    let b_all_zeros = b.iter().all(|&v| v == 0.0);

    if b_all_zeros {
      return Ok(Array1::ones(V.len()));
    }

    Ok(((a + &b - 1.0) / b).powf(-1.0 / theta))
  }

  fn partial_derivative(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    let U = X.column(0);
    let V = X.column(1);

    let theta = self.theta.unwrap();
    let A = V.powf(-theta - 1.0);

    if A.is_all_infinite() {
      return Ok(Array1::zeros(V.len()));
    }

    let B = V.powf(-theta) + U.powf(-theta) - 1.0;
    let h = B.powf((-1.0 - theta) / theta);
    Ok(A * h)
  }

  fn compute_theta(&self) -> f64 {
    if self.tau.is_some() && self.tau.unwrap() == 1.0 {
      return f64::INFINITY;
    }

    let tau = self.tau.unwrap();

    2.0 * tau / (1.0 - tau)
  }
}
