use impl_new_derive::ImplNew;
use ndarray::{s, Array1};
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

#[derive(ImplNew)]
pub struct BM<T> {
  pub n: usize,
  pub t: Option<T>,
  pub m: Option<usize>,
}

#[cfg(feature = "f64")]
impl SamplingExt<f64> for BM<f64> {
  fn sample(&self) -> Array1<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());
    let mut bm = Array1::<f64>::zeros(self.n);
    bm.slice_mut(s![1..]).assign(&gn);

    for i in 1..self.n {
      bm[i] += bm[i - 1];
    }

    bm
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

#[cfg(feature = "f32")]
impl SamplingExt<f32> for BM<f32> {
  fn sample(&self) -> Array1<f32> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn = Array1::random(self.n - 1, Normal::new(0.0, dt.sqrt()).unwrap());
    let mut bm = Array1::<f32>::zeros(self.n);
    bm.slice_mut(s![1..]).assign(&gn);

    for i in 1..self.n {
      bm[i] += bm[i - 1];
    }

    bm
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f32> {
    use crate::stats::distr::normal::SimdNormal;

    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let gn = Array1::random(self.n - 1, SimdNormal::new(0.0, dt.sqrt()));
    let mut bm = Array1::<f32>::zeros(self.n);
    bm.slice_mut(s![1..]).assign(&gn);

    for i in 1..self.n {
      bm[i] += bm[i - 1];
    }

    bm
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
