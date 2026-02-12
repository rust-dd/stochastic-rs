use core::f64;
use std::cmp::Ordering;
use std::error::Error;

use ndarray::stack;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
use rand_distr::Uniform;
use roots::find_root_brent;
use roots::SimpleConvergency;

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

    let v = Array1::<f64>::random(n, Uniform::new(0.0, 1.0)?);
    let c = Array1::<f64>::random(n, Uniform::new(0.0, 1.0)?);
    let u = self.percent_point(&c, &v)?;

    Ok(stack![Axis(1), u, v])
  }

  fn fit(&mut self, X: &Array2<f64>) -> Result<(), Box<dyn Error>> {
    let U = X.column(0).to_owned();
    let V = X.column(1).to_owned();

    self.check_marginal(&U)?;
    self.check_marginal(&V)?;

    let (tau, ..) = kendalls::tau_b_with_comparator(&U.to_vec(), &V.to_vec(), |a, b| {
      a.partial_cmp(b).unwrap_or(Ordering::Greater)
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

  fn pdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>>;

  fn log_pdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    Ok(self.pdf(X)?.ln())
  }

  fn cdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>>;

  fn percent_point(&self, y: &Array1<f64>, V: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    let n = y.len();
    let mut results = Array1::zeros(n);

    for i in 0..n {
      let y_i = y[i];
      let v_i = V[i];

      let f = |u| self.partial_derivative_scalar(u, v_i).unwrap() - y_i;
      let mut convergency = SimpleConvergency {
        eps: f64::EPSILON,
        max_iter: 50,
      };
      let min = find_root_brent(f64::EPSILON, 1.0, f, &mut convergency);
      results[i] = min.unwrap_or(f64::EPSILON);
    }

    Ok(results)
  }

  fn ppf(&self, y: &Array1<f64>, V: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.percent_point(y, V)
  }

  fn partial_derivative(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    let n = X.nrows();
    let mut X_prime = X.clone();
    let mut delta = Array1::zeros(n);
    for i in 0..n {
      delta[i] = if X[[i, 1]] > 0.5 { -0.0001 } else { 0.0001 };
      X_prime[[i, 1]] = X[[i, 1]] + delta[i];
    }

    let f = self.cdf(X).unwrap();
    let f_prime = self.cdf(&X_prime).unwrap();

    let mut deriv = Array1::zeros(n);
    for i in 0..n {
      deriv[i] = (f_prime[i] - f[i]) / delta[i];
    }

    Ok(deriv)
  }

  fn partial_derivative_scalar(&self, U: f64, V: f64) -> Result<f64, Box<dyn Error>> {
    self.check_fit()?;
    let X = stack![Axis(1), Array1::from(vec![U]), Array1::from(vec![V])];
    let out = self.partial_derivative(&X);

    Ok(*out?.get(0).unwrap())
  }
}
