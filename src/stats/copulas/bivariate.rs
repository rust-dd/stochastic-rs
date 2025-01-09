use std::{cmp::Ordering, error::Error};

use ndarray::{stack, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
use rand_distr::Uniform;

pub mod clayton;
pub mod frank;
pub mod gumbel;
pub mod independence;

#[derive(Debug, Clone, Copy)]
pub enum CopulaType {
  Clayton,
  Frank,
  Gumbel,
  Independence,
}

pub trait Bivariate {
  fn r#type(&self) -> CopulaType;

  fn tau(&self) -> Option<f64>;

  fn set_tau(&mut self, tau: f64);

  fn theta(&self) -> Option<f64>;

  fn theta_bounds(&self) -> (f64, f64);

  fn invalid_thetas(&self) -> Vec<f64>;

  fn set_theta(&mut self, theta: f64);

  fn check_theta(&self) -> Result<(), String> {
    let (lower, upper) = self.theta_bounds();
    let theta = self.theta().unwrap();
    let invalid = self.invalid_thetas();

    if !(lower <= theta && theta <= upper) || invalid.contains(&theta) {
      return Err(format!(
        "Theta must be in the interval [{}, {}] and not in {:?}",
        lower, upper, invalid
      ));
    }

    Ok(())
  }

  fn compute_theta(&self) -> f64;

  fn _compute_theta(&mut self) {
    self.set_theta(self.compute_theta());
    let _ = self.check_theta();
  }

  fn generator(&self, t: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>>;

  fn sample(&mut self, n: usize) -> Result<Array2<f64>, Box<dyn Error>> {
    if self.tau().is_none() {
      return Err("Tau is not defined".into());
    }

    let tau = self.tau().unwrap();

    if !(-1.0..1.0).contains(&tau) {
      return Err("Tau must be in the interval (-1, 1)".into());
    }

    let v = Array1::<f64>::random(n, Uniform::new(0.0, 1.0));
    let c = Array1::<f64>::random(n, Uniform::new(0.0, 1.0));
    let u = self.percent_point(&c, &v)?;

    Ok(stack![Axis(1), u, v])
  }

  fn fit(&mut self, X: &Array2<f64>) -> Result<(), Box<dyn Error>> {
    let U = X.column(0).to_owned();
    let V = X.column(1).to_owned();

    self.check_marginal(&U)?;
    self.check_marginal(&V)?;

    let (tau, ..) = kendalls::tau_b_with_comparator(&U.to_vec(), &V.to_vec(), |a, b| {
      a.partial_cmp(&b).unwrap_or(Ordering::Greater)
    })?;

    self.set_tau(tau);
    self._compute_theta();

    Ok(())
  }

  fn check_fit(&self) -> Result<(), Box<dyn Error>> {
    if self.theta().is_none() {
      return Err("Fit the copula first".into());
    }

    self.check_theta()?;
    Ok(())
  }

  fn check_marginal(&self, u: &Array1<f64>) -> Result<(), String> {
    if !(0.0..=1.0).contains(u.min().unwrap()) || !(0.0..=1.0).contains(u.max().unwrap()) {
      return Err("Marginal values must be in the interval [0, 1]".into());
    }

    // sort
    let mut empirical_cdf = u.to_vec();
    empirical_cdf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Greater));
    let empirical_cdf = Array1::from(empirical_cdf);
    let uniform = Array1::linspace(0.0, 1.0, u.len());
    let ks = (empirical_cdf - uniform).mapv(f64::abs);
    let ks = ks.max().unwrap();

    if *ks > 1.627 / (u.len() as f64).sqrt() {
      return Err("Marginal values do not follow a uniform distribution".into());
    }

    Ok(())
  }

  fn probability_density(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>>;

  fn pdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.probability_density(X)
  }

  fn log_probability_density(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    let pdf = self.probability_density(X)?;
    let log_pdf = pdf.mapv(|val| (val + 1e-32).ln());
    Ok(log_pdf)
  }

  fn cumulative_distribution(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>>;

  fn cdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.cumulative_distribution(X)
  }

  fn percent_point(&self, y: &Array1<f64>, V: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    todo!()
  }

  fn ppf(&self, y: &Array1<f64>, V: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.percent_point(y, V)
  }

  fn partial_derivative(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    todo!()
  }

  fn partial_derivative_scalar(
    &self,
    u: Array1<f64>,
    v: Array1<f64>,
  ) -> Result<f64, Box<dyn Error>> {
    todo!()
  }
}
