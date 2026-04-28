//! # Vg
//!
//! $$
//! X_t=\theta G_t+\sigma W_{G_t},\quad G_t\sim\Gamma(\nu^{-1}t,\nu)
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct Vg<T: FloatExt, S: SeedExt = Unseeded> {
  /// Drift / long-run mean-level parameter.
  pub mu: T,
  /// Diffusion / noise scale parameter.
  pub sigma: T,
  /// Volatility-of-volatility / tail-thickness parameter.
  pub nu: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  pub seed: S,
}

impl<T: FloatExt> Vg<T> {
  pub fn new(mu: T, sigma: T, nu: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    assert!(nu > T::zero(), "nu must be positive");
    Self {
      mu,
      sigma,
      nu,
      n,
      x0,
      t,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> Vg<T, Deterministic> {
  pub fn seeded(mu: T, sigma: T, nu: T, n: usize, x0: Option<T>, t: Option<T>, seed: u64) -> Self {
    assert!(nu > T::zero(), "nu must be positive");
    Self {
      mu,
      sigma,
      nu,
      n,
      x0,
      t,
      seed: Deterministic::new(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> Vg<T, S> {
  #[inline]
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Vg<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut vg = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return vg;
    }
    vg[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return vg;
    }

    let dt = self.dt();
    let gamma = SimdGamma::from_seed_source(dt / self.nu, self.nu, &self.seed);
    let mut gammas = Array1::<T>::zeros(self.n - 1);
    gamma.fill_slice_fast(gammas.as_slice_mut().unwrap());
    let normal = SimdNormal::<T>::from_seed_source(T::zero(), T::one(), &self.seed);
    let mut z = Array1::<T>::zeros(self.n - 1);
    normal.fill_slice_fast(z.as_slice_mut().unwrap());

    for i in 1..self.n {
      vg[i] = vg[i - 1] + self.mu * gammas[i - 1] + self.sigma * gammas[i - 1].sqrt() * z[i - 1];
    }

    vg
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::ProcessExt;

  #[test]
  fn n_eq_1_keeps_initial_value() {
    let p = Vg::new(0.1_f64, 0.2, 0.3, 1, Some(2.5), Some(1.0));
    let x = p.sample();
    assert_eq!(x.len(), 1);
    assert_eq!(x[0], 2.5);
  }
}

py_process_1d!(PyVg, Vg,
  sig: (mu, sigma, nu, n, x0=None, t=None, seed=None, dtype=None),
  params: (mu: f64, sigma: f64, nu: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
