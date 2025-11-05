use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

#[derive(ImplNew)]
pub struct Gn<T> {
  pub n: usize,
  pub t: Option<T>,
  pub m: Option<usize>,
}

impl SamplingExt<f64> for Gn<f64> {
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / self.n as f64;
    Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap())
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(feature = "f32")]
impl SamplingExt<f32> for Gn<f32> {
  fn sample(&self) -> Array1<f32> {
    let dt = self.t.unwrap_or(1.0) / self.n as f32;
    Array1::random(self.n, Normal::new(0.0, (dt.sqrt()) as f64).unwrap()).mapv(|x| x as f32)
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
