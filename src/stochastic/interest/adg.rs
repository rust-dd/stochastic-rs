use impl_new_derive::ImplNew;
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingVExt;

/// Ahn-Dittmar-Gallant (ADG) model
#[derive(ImplNew)]
pub struct ADG<T> {
  pub k: fn(T) -> T,
  pub theta: fn(T) -> T,
  pub sigma: Array1<T>,
  pub phi: fn(T) -> T,
  pub b: fn(T) -> T,
  pub c: fn(T) -> T,
  pub n: usize,
  pub xn: usize,
  pub x0: Array1<T>,
  pub t: Option<T>,
  pub m: Option<usize>,
}

impl SamplingVExt<f64> for ADG<f64> {
  fn sample(&self) -> Array2<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;

    let mut adg = Array2::<f64>::zeros((self.xn, self.n));
    for i in 0..self.xn {
      adg[(i, 0)] = self.x0[i];
    }

    for i in 0..self.xn {
      let gn = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());

      for j in 1..self.n {
        let t = j as f64 * dt;
        adg[(i, j)] = adg[(i, j - 1)]
          + ((self.k)(t) - (self.theta)(t) * adg[(i, j - 1)]) * dt
          + self.sigma[i] * gn[j - 1];
      }
    }

    let mut r = Array2::zeros((self.xn, self.n));

    for i in 0..self.xn {
      let phi = Array1::<f64>::from_shape_fn(self.n, |j| (self.phi)(j as f64 * dt));
      let b = Array1::<f64>::from_shape_fn(self.n, |j| (self.b)(j as f64 * dt));
      let c = Array1::<f64>::from_shape_fn(self.n, |j| (self.c)(j as f64 * dt));

      r.row_mut(i)
        .assign(&(phi + b * adg.row(i).t().to_owned() * c * adg.row(i)));
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
