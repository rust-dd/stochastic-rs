use std::error::Error;

use ndarray::Array1;
use ndarray::Array2;

use super::Bivariate;
use super::CopulaType;

#[derive(Debug, Clone)]
pub struct Gumbel {
  pub r#type: CopulaType,
  pub theta: Option<f64>,
  pub tau: Option<f64>,
  pub theta_bounds: (f64, f64),
  pub invalid_thetas: Vec<f64>,
}

impl Gumbel {
  pub fn new(theta: Option<f64>, tau: Option<f64>) -> Self {
    Self {
      r#type: CopulaType::Gumbel,
      theta,
      tau,
      theta_bounds: (1.0, f64::INFINITY),
      invalid_thetas: vec![0.0],
    }
  }
}

impl Bivariate for Gumbel {
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
    Ok((-t.ln()).powf(self.theta.unwrap()))
  }

  fn pdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    let U = X.column(0);
    let V = X.column(1);

    let theta = self.theta.unwrap();

    if theta == 1.0 {
      return Ok(Array1::ones(U.len()));
    }

    let a = (&U * &V).powf(-1.0);
    let tmp = (-U.ln()).powf(theta) + (-V.ln()).powf(theta);
    let b = tmp.powf(-2.0 + 2.0 / theta);
    let c = (U.ln() * V.ln()).powf(theta - 1.0);
    let d = 1.0 + (theta - 1.0) * tmp.powf(-1.0 / theta);
    let out = self.cdf(&X)? * a * b * c * d;

    Ok(out)
  }

  fn cdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    let U = X.column(0);
    let V = X.column(1);

    let theta = self.theta.unwrap();

    if theta == 1.0 {
      return Ok(&U * &V);
    }

    let h = (-U.ln()).powf(theta) + (-V.ln()).powf(theta);
    let h = -h.powf(1.0 / theta);
    let cdfs = h.exp();

    Ok(cdfs)
  }

  fn percent_point(&self, y: &Array1<f64>, V: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.check_fit()?;

    if self.theta.unwrap() == 1.0 {
      return Ok(y.to_owned());
    }

    let out = Bivariate::percent_point(self, y, V)?;
    Ok(out)
  }

  fn partial_derivative(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    self.check_fit()?;

    let U = X.column(0);
    let V = X.column(1);

    let theta = self.theta.unwrap();

    if theta == 1.0 {
      return Ok(V.to_owned());
    }

    let t1 = (-U.ln()).powf(theta);
    let t2 = (-V.ln()).powf(theta);
    let p1 = self.cdf(X)?;
    let p2 = (t1 + t2).powf(-1.0 + 1.0 / theta);
    let p3 = (-V.ln()).powf(theta - 1.0);
    let out = p1 * p2 * p3 / V;

    Ok(out)
  }

  fn compute_theta(&self) -> f64 {
    let tau = self.tau.unwrap();

    if tau == 1.0 {
      panic!("Tau cannot be 1.0 for Gumbel copula");
    }

    1.0 / (1.0 - tau)
  }
}
