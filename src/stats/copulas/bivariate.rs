use std::error::Error;

use ndarray::{stack, Array1, Array2, Axis};

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

  fn sample(&mut self, n_samples: usize) -> Result<Array2<f64>, Box<dyn Error>>;

  fn fit(&mut self, X: &Array2<f64>) -> Result<(), Box<dyn Error>>;

  fn check_fit(&self) -> Result<(), Box<dyn Error>> {
    if self.theta().is_none() {
      return Err("Fit the copula first".into());
    }

    self.check_theta()?;
    Ok(())
  }

  fn check_marginal(&self, u: &Array1<f64>) -> Result<(), String>;

  fn probability_density(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>>;

  fn log_probability_density(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    let pdf = self.probability_density(X)?;
    let out = pdf.mapv(|val| (val + 1e-32).ln());
    Ok(out)
  }

  fn pdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.probability_density(X)
  }

  fn cumulative_distribution(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>>;

  fn cdf(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.cumulative_distribution(X)
  }

  fn percent_point(&self, y: &Array1<f64>, V: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>>;

  fn ppf(&self, y: &Array1<f64>, V: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    self.percent_point(y, V)
  }

  fn partial_derivative(&self, X: &Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>>;

  fn partial_derivative_scalar(
    &self,
    u: Array1<f64>,
    v: Array1<f64>,
  ) -> Result<f64, Box<dyn Error>> {
    self.check_fit()?;

    // X = np.column_stack((U, V))
    let arr = stack![Axis(1), u, v];
    let pd = self.partial_derivative(&arr)?;
    Ok(pd[0])
  }
}
