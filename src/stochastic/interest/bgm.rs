//! A simple single-factor BGM (Brace–Gatarek–Musiela) model for forward rates.
//!
//! Produces a 2D array (`(xn, n)`) of forward rate paths. Each row represents a separate
//! forward rate, evolving over `n` time steps.
//!
//! # Parameters
//! - `lambda`: The drift/volatility multiplier for each forward rate.
//! - `x0`: Initial forward rates for each path.
//! - `xn`: Number of forward rates (rows) to simulate.
//! - `t`: Total time horizon.
//! - `n`: Number of time steps in the simulation.
//! - `m`: Batch size for parallel sampling (if used).

use crate::stochastic::SamplingVExt;
use impl_new_derive::ImplNew;
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand_distr::Normal;

#[derive(ImplNew)]
pub struct BGM<T> {
  /// Drift/volatility multiplier for each forward rate.
  pub lambda: Array1<T>,
  /// Initial forward rates for each path.
  pub x0: Array1<T>,
  /// Number of forward rates (rows) to simulate.
  pub xn: usize,
  /// Total time horizon.
  pub t: Option<T>,
  /// Number of time steps in the simulation.
  pub n: usize,
  /// Batch size for parallel sampling (if used).
  pub m: Option<usize>,
}

impl SamplingVExt<f64> for BGM<f64> {
  fn sample(&self) -> Array2<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let mut fwd = Array2::<f64>::zeros((self.xn, self.n));

    for i in 0..self.xn {
      fwd[(i, 0)] = self.x0[i];
    }

    for i in 0..self.xn {
      let gn = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());
      for j in 1..self.n {
        let f_old = fwd[(i, j - 1)];
        fwd[(i, j)] = f_old + f_old * self.lambda[i] * gn[j - 1];
      }
    }

    fwd
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(feature = "f32")]
impl SamplingVExt<f32> for BGM<f32> {
  fn sample(&self) -> Array2<f32> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let mut fwd = Array2::<f32>::zeros((self.xn, self.n));

    for i in 0..self.xn {
      fwd[(i, 0)] = self.x0[i];
    }

    for i in 0..self.xn {
      let gn = Array1::random(self.n, Normal::new(0.0, (dt.sqrt()) as f64).unwrap()).mapv(|x| x as f32);
      for j in 1..self.n {
        let f_old = fwd[(i, j - 1)];
        fwd[(i, j)] = f_old + f_old * self.lambda[i] * gn[j - 1];
      }
    }

    fwd
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
