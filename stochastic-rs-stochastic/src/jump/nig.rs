//! # Nig
//!
//! $$
//! X_t=\mu t+\beta I_t+W_{I_t},\quad I_t\sim\mathrm{Ig}(\delta t,\gamma)
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::inverse_gauss::SimdInverseGauss;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct Nig<T: FloatExt, S: SeedExt = Unseeded> {
  /// Long-run target level / model location parameter.
  pub theta: T,
  /// Diffusion / noise scale parameter.
  pub sigma: T,
  /// Mean-reversion speed parameter.
  pub kappa: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Nig<T, S> {
  pub fn new(theta: T, sigma: T, kappa: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    assert!(kappa > T::zero(), "kappa must be positive");
    Self {
      theta,
      sigma,
      kappa,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> Nig<T, S> {
  #[inline]
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Nig<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut nig = Array1::zeros(self.n);
    if self.n == 0 {
      return nig;
    }
    nig[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return nig;
    }

    let dt = self.dt();
    // For Nig: G_dt ~ Ig(mean=dt, shape=dt^2/kappa).
    let shape = dt * dt / self.kappa;
    let ig_dist = SimdInverseGauss::from_seed_source(dt, shape, &self.seed);
    let mut ig = Array1::<T>::zeros(self.n - 1);
    ig_dist.fill_slice_fast(ig.as_slice_mut().unwrap());
    let normal = SimdNormal::<T>::from_seed_source(T::zero(), T::one(), &self.seed);
    let mut z = Array1::<T>::zeros(self.n - 1);
    normal.fill_slice_fast(z.as_slice_mut().unwrap());

    for i in 1..self.n {
      nig[i] = nig[i - 1] + self.theta * ig[i - 1] + self.sigma * ig[i - 1].sqrt() * z[i - 1]
    }

    nig
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::ProcessExt;

  #[test]
  fn n_eq_1_keeps_initial_value() {
    let p = Nig::new(0.1_f64, 0.2, 0.3, 1, Some(4.0), Some(1.0), Unseeded);
    let x = p.sample();
    assert_eq!(x.len(), 1);
    assert_eq!(x[0], 4.0);
  }
}

py_process_1d!(PyNig, Nig,
  sig: (theta, sigma, kappa, n, x0=None, t=None, seed=None, dtype=None),
  params: (theta: f64, sigma: f64, kappa: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
