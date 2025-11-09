//! A multi-dimensional Wu–Zhang-style model, combining a forward rate and a stochastic
//! volatility process for each dimension.
//!
//! Produces a 2D array (`(2*xn, n)`) of forward rates and volatilities. The first `xn` rows
//! are the forward rates; the next `xn` rows are the volatilities, each evolving over `n` steps.
//!
//! # Parameters
//! - `alpha`: Mean reversion level for each dimension's volatility.
//! - `beta`: Mean reversion speed for each dimension's volatility.
//! - `nu`: Volatility-of-volatility parameter for each dimension.
//! - `lambda`: Parameter controlling how volatility impacts the forward rate.
//! - `x0`: Initial forward rates for each dimension.
//! - `v0`: Initial volatilities for each dimension.
//! - `xn`: Number of `(rate, vol)` pairs.
//! - `t`: Total time horizon.
//! - `n`: Number of time steps in the simulation.
//! - `m`: Batch size for parallel sampling (if used).

use crate::stochastic::SamplingVExt;
use impl_new_derive::ImplNew;
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand_distr::Normal;

#[derive(ImplNew)]
pub struct WuZhangD<T> {
  /// Mean reversion level for each dimension's volatility.
  pub alpha: Array1<T>,
  /// Mean reversion speed for each dimension's volatility.
  pub beta: Array1<T>,
  /// Volatility of volatility for each dimension.
  pub nu: Array1<T>,
  /// Parameter controlling the impact of volatility on the forward rate.
  pub lambda: Array1<T>,
  /// Initial forward rates for each dimension.
  pub x0: Array1<T>,
  /// Initial volatilities for each dimension.
  pub v0: Array1<T>,
  /// Number of (rate, vol) pairs.
  pub xn: usize,
  /// Total time horizon.
  pub t: Option<T>,
  /// Number of time steps in the simulation.
  pub n: usize,
  /// Batch size for parallel sampling (if used).
  pub m: Option<usize>,
}

#[cfg(feature = "f64")]
impl SamplingVExt<f64> for WuZhangD<f64> {
  fn sample(&self) -> Array2<f64> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let mut fv = Array2::<f64>::zeros((2 * self.xn, self.n));

    for i in 0..self.xn {
      fv[(i, 0)] = self.x0[i];
      fv[(i + self.xn, 0)] = self.v0[i];
    }

    for i in 0..self.xn {
      let gn_f = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());
      let gn_v = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());

      for j in 1..self.n {
        let v_old = fv[(i + self.xn, j - 1)].max(0.0);
        let f_old = fv[(i, j - 1)].max(0.0);

        let dv =
          (self.alpha[i] - self.beta[i] * v_old) * dt + self.nu[i] * v_old.sqrt() * gn_v[j - 1];

        let v_new = (v_old + dv).max(0.0);
        fv[(i + self.xn, j)] = v_new;

        let df = f_old * self.lambda[i] * v_new.sqrt() * gn_f[j - 1];
        fv[(i, j)] = f_old + df;
      }
    }

    fv
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(feature = "f32")]
impl SamplingVExt<f32> for WuZhangD<f32> {
  fn sample(&self) -> Array2<f32> {
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let mut fv = Array2::<f32>::zeros((2 * self.xn, self.n));

    for i in 0..self.xn {
      fv[(i, 0)] = self.x0[i];
      fv[(i + self.xn, 0)] = self.v0[i];
    }

    for i in 0..self.xn {
      let gn_f = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());
      let gn_v = Array1::random(self.n, Normal::new(0.0, dt.sqrt()).unwrap());

      for j in 1..self.n {
        let v_old = fv[(i + self.xn, j - 1)].max(0.0);
        let f_old = fv[(i, j - 1)].max(0.0);

        let dv =
          (self.alpha[i] - self.beta[i] * v_old) * dt + self.nu[i] * v_old.sqrt() * gn_v[j - 1];

        let v_new = (v_old + dv).max(0.0);
        fv[(i + self.xn, j)] = v_new;

        let df = f_old * self.lambda[i] * v_new.sqrt() * gn_f[j - 1];
        fv[(i, j)] = f_old + df;
      }
    }

    fv
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
