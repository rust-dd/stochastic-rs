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

use ndarray::Array1;
use ndarray::Array2;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::ProcessExt;

pub struct BGM<T: Float> {
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
  gn: Gn<T>,
}

impl<T: Float> BGM<T> {
  pub fn new(lambda: Array1<T>, x0: Array1<T>, xn: usize, t: Option<T>, n: usize) -> Self {
    Self {
      lambda,
      x0,
      xn,
      t,
      n,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: Float> ProcessExt<T> for BGM<T> {
  type Output = Array2<T>;

  fn sample(&self) -> Self::Output {
    let mut fwd = Array2::<T>::zeros((self.xn, self.n));

    for i in 0..self.xn {
      fwd[(i, 0)] = self.x0[i];
    }

    for i in 0..self.xn {
      let gn = &self.gn.sample();

      for j in 1..self.n {
        let f_old = fwd[(i, j - 1)];
        fwd[(i, j)] = f_old + f_old * self.lambda[i] * gn[j - 1];
      }
    }

    fwd
  }
}
