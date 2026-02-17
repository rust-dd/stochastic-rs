//! # IG
//!
//! $$
//! L_t\sim\mathrm{IG}(\mu t,\lambda t),\quad X_t=L_t\text{ or time-change driver}
//! $$
//!
use ndarray::Array1;
use ndarray_rand::RandomExt;

use crate::distributions::inverse_gauss::SimdInverseGauss;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct IG<T: FloatExt> {
  /// Model asymmetry / nonlinearity parameter.
  pub gamma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
}

impl<T: FloatExt> IG<T> {
  pub fn new(gamma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    assert!(gamma > T::zero(), "gamma must be positive");
    Self { gamma, n, x0, t }
  }

  #[inline]
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }
}

impl<T: FloatExt> ProcessExt<T> for IG<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut ig = Array1::zeros(self.n);
    if self.n == 0 {
      return ig;
    }
    ig[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return ig;
    }

    let dt = self.dt();
    // Single-parameter IG subordinator:
    // increments are strictly positive and independent over grid steps.
    let mean = self.gamma * dt;
    let shape = mean * mean;
    let ig_dist = SimdInverseGauss::new(mean, shape);
    let inc = Array1::random(self.n - 1, &ig_dist);

    for i in 1..self.n {
      ig[i] = ig[i - 1] + inc[i - 1];
    }

    ig
  }
}

py_process_1d!(PyIG, IG,
  sig: (gamma_, n, x0=None, t=None, dtype=None),
  params: (gamma_: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::ProcessExt;

  #[test]
  fn ig_path_is_non_decreasing() {
    let p = IG::new(1.0_f64, 256, Some(0.0), Some(1.0));
    let x = p.sample();
    assert_eq!(x.len(), 256);
    assert!(x.windows(2).into_iter().all(|w| w[1] >= w[0]));
  }

  #[test]
  fn n_eq_1_keeps_initial_value() {
    let p = IG::new(1.0_f64, 1, Some(3.5), Some(1.0));
    let x = p.sample();
    assert_eq!(x.len(), 1);
    assert_eq!(x[0], 3.5);
  }
}