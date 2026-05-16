//! # Ig
//!
//! $$
//! L_t\sim\mathrm{Ig}(\mu t,\lambda t),\quad X_t=L_t\text{ or time-change driver}
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::inverse_gauss::SimdInverseGauss;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct Ig<T: FloatExt, S: SeedExt = Unseeded> {
  /// Model asymmetry / nonlinearity parameter.
  pub gamma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Ig<T, S> {
  pub fn new(gamma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    assert!(gamma > T::zero(), "gamma must be positive");
    Self {
      gamma,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> Ig<T, S> {
  #[inline]
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Ig<T, S> {
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
    // Single-parameter Ig subordinator:
    // increments are strictly positive and independent over grid steps.
    let mean = self.gamma * dt;
    let shape = mean * mean;
    let ig_dist = SimdInverseGauss::from_seed_source(mean, shape, &self.seed);
    let mut inc = Array1::<T>::zeros(self.n - 1);
    ig_dist.fill_slice_fast(inc.as_slice_mut().unwrap());

    for i in 1..self.n {
      ig[i] = ig[i - 1] + inc[i - 1];
    }

    ig
  }
}

py_process_1d!(PyIg, Ig,
  sig: (gamma_, n, x0=None, t=None, seed=None, dtype=None),
  params: (gamma_: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::ProcessExt;

  #[test]
  fn ig_path_is_non_decreasing() {
    let p = Ig::new(1.0_f64, 256, Some(0.0), Some(1.0), Unseeded);
    let x = p.sample();
    assert_eq!(x.len(), 256);
    assert!(x.windows(2).into_iter().all(|w| w[1] >= w[0]));
  }

  #[test]
  fn n_eq_1_keeps_initial_value() {
    let p = Ig::new(1.0_f64, 1, Some(3.5), Some(1.0), Unseeded);
    let x = p.sample();
    assert_eq!(x.len(), 1);
    assert_eq!(x[0], 3.5);
  }
}
