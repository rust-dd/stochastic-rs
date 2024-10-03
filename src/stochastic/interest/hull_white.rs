use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::Sampling;

/// Hull-White process.
/// dX(t) = theta(t)dt - alpha * X(t)dt + sigma * dW(t)
/// where X(t) is the Hull-White process.
pub struct HullWhite {
  pub theta: fn(f64) -> f64,
  pub alpha: f64,
  pub sigma: f64,
  pub n: usize,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
}

impl HullWhite {
  #[must_use]
  pub fn new(params: &Self) -> Self {
    Self {
      theta: params.theta,
      alpha: params.alpha,
      sigma: params.sigma,
      n: params.n,
      x0: params.x0,
      t: params.t,
      m: params.m,
    }
  }
}

impl Sampling<f64> for HullWhite {
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / self.n as f64;
    let gn = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut hw = Array1::<f64>::zeros(self.n + 1);
    hw[0] = self.x0.unwrap_or(0.0);

    for i in 1..=self.n {
      hw[i] = hw[i - 1]
        + ((self.theta)(i as f64 * dt) - self.alpha * hw[i - 1]) * dt
        + self.sigma * gn[i - 1]
    }

    hw
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}