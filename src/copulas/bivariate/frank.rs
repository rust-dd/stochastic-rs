use core::f64;
use std::error::Error;

use gauss_quad::GaussLegendre;
use ndarray::Array1;
use ndarray::Array2;

use crate::copulas::bivariate::BivariateExt;
use crate::copulas::bivariate::CopulaType;

#[derive(Debug, Clone)]
pub struct Frank {
  pub r#type: CopulaType,
  pub theta: Option<f64>,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Frank {
  pub fn new(theta: Option<f64>, tau: Option<f64>) -> Self {
    Self {
      r#type: CopulaType::Frank,
      theta,
      tau,
      theta_bounds: (f64::NEG_INFINITY, f64::INFINITY),
      invalid_thetas: vec![0.0],
    }
  }
}

impl BivariateExt for Frank {
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
    let theta = self.theta.unwrap();
    let a = ((-theta * t).exp() - 1.0) / ((-theta).exp() - 1.0);
    let out = -(a.ln());
    Ok(out)
  }

  fn pdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    let U = X.column(0).to_owned();
    let V = X.column(1).to_owned();

    let theta = self.theta.unwrap();

    if theta == 0.0 {
      return Ok(Array1::ones(U.len()));
    }

    let num = (-theta * self._g(&Array1::ones(U.len()))?) * (1.0 + self._g(&(&U + &V))?);
    let aux = self._g(&U)? + self._g(&V)? + self._g(&Array1::ones(U.len()))?;
    let den = aux.pow2();
    Ok(num / den)
  }

  fn cdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    let U = X.column(0);
    let V = X.column(1);

    let theta = self.theta.unwrap();
    let num = ((-theta * &U).exp() - 1.0) * ((-theta * &V).exp() - 1.0);
    let den = (-theta).exp() - 1.0;
    let out = -1.0 / theta * (1.0 + num / den).ln();
    Ok(out)
  }

  #[allow(clippy::only_used_in_recursion)]
  fn percent_point(&self, y: &Array1<f64>, V: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    let theta = self.theta.unwrap();

    if theta == 0.0 {
      return Ok(V.clone());
    }

    let out = BivariateExt::percent_point(self, y, V)?;
    Ok(out)
  }

  fn partial_derivative(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    self.check_fit()?;

    let U = X.column(0).to_owned();
    let V = X.column(1).to_owned();

    let theta = self.theta.unwrap();

    if theta == 0.0 {
      return Ok(V.clone());
    }

    let num = self._g(&U)? * self._g(&V)? + self._g(&U)?;
    let den = self._g(&U)? + self._g(&V)? + self._g(&Array1::ones(U.len()))?;
    Ok(num / den)
  }

  fn compute_theta(&self) -> f64 {
    self
      .least_squares(Self::_tau_to_theta, 1.0, f64::MIN.ln(), f64::MAX.ln())
      .unwrap_or(1.0)
  }
}

impl Frank {
  fn _g(&self, z: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    Ok((-self.theta.unwrap() * z).exp() - 1.0)
  }

  fn _tau_to_theta(tau: f64, alpha: f64) -> f64 {
    let integrand = |u: f64| u / (u.exp() - 1.0);
    let quad = GaussLegendre::new(5).unwrap();
    let integral = quad.integrate(f64::EPSILON, alpha, integrand);
    4.0 * (integral - 1.0) / alpha + 1.0 - tau
  }

  // TODO: Improve this implementation
  fn least_squares<F>(
    &self,
    f: F,
    initial_guess: f64,
    lower_bound: f64,
    upper_bound: f64,
  ) -> Option<f64>
  where
    F: Fn(f64, f64) -> f64,
  {
    let mut guess = initial_guess;
    let tol = 1e-6;
    for _ in 0..1000 {
      let v = f(self.tau.unwrap(), guess);
      if v.abs() < tol {
        return Some(guess);
      }
      guess -= v * 0.01;
      if guess < lower_bound || guess > upper_bound {
        return None;
      }
    }
    None
  }
}
