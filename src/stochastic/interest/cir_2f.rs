use impl_new_derive::ImplNew;
use ndarray::Array1;

use super::cir::CIR;
use crate::stochastic::SamplingExt;

#[derive(ImplNew)]
pub struct CIR2F<T> {
  pub x: CIR<T>,
  pub y: CIR<T>,
  pub phi: fn(T) -> T,
}

#[cfg(feature = "f64")]
impl SamplingExt<f64> for CIR2F<f64> {
  fn sample(&self) -> Array1<f64> {
    let x = self.x.sample();
    let y = self.y.sample();

    let dt = self.x.t.unwrap_or(1.0) / (self.n() - 1) as f64;
    let phi = Array1::<f64>::from_shape_fn(self.n(), |i| (self.phi)(i as f64 * dt));

    x + y * phi
  }

  fn n(&self) -> usize {
    self.x.n()
  }

  fn m(&self) -> Option<usize> {
    self.x.m()
  }
}

#[cfg(feature = "f32")]
impl SamplingExt<f32> for CIR2F<f32> {
  fn sample(&self) -> Array1<f32> {
    let x = self.x.sample();
    let y = self.y.sample();

    let dt = self.x.t.unwrap_or(1.0) / (self.n() - 1) as f32;
    let phi = Array1::<f32>::from_shape_fn(self.n(), |i| (self.phi)(i as f32 * dt));

    x + y * phi
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f32> {
    let x = self.x.sample_simd();
    let y = self.y.sample_simd();

    let dt = self.x.t.unwrap_or(1.0) / (self.n() - 1) as f32;
    let phi = Array1::<f32>::from_shape_fn(self.n(), |i| (self.phi)(i as f32 * dt));

    x + y * phi
  }

  fn n(&self) -> usize {
    self.x.n()
  }

  fn m(&self) -> Option<usize> {
    self.x.m()
  }
}
