use std::sync::Arc;

use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

#[allow(non_snake_case)]
#[derive(ImplNew)]
pub struct HoLee<T> {
  pub f_T: Option<Arc<dyn Fn(T) -> T + Send + Sync + 'static>>,
  pub theta: Option<T>,
  pub sigma: T,
  pub n: usize,
  pub t: T,
  pub m: Option<usize>,
}

impl SamplingExt<f64> for HoLee<f64> {
  fn sample(&self) -> Array1<f64> {
    assert!(
      self.theta.is_none() && self.f_T.is_none(),
      "theta or f_T must be provided"
    );
    let dt = self.t / (self.n - 1) as f64;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());

    let mut r = Array1::<f64>::zeros(self.n);

    for i in 1..self.n {
      let drift = if let Some(r#fn) = self.f_T.as_ref() {
        (r#fn)(i as f64 * dt) + self.sigma.powf(2.0)
      } else {
        self.theta.unwrap() + self.sigma.powf(2.0)
      };

      r[i] = r[i - 1] + drift * dt + self.sigma * gn[i - 1];
    }

    r
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }
}
