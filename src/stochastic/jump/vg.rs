//! # Vg
//!
//! $$
//! X_t=\theta G_t+\sigma W_{G_t},\quad G_t\sim\Gamma(\nu^{-1}t,\nu)
//! $$
//!
use ndarray::Array1;
use ndarray_rand::RandomExt;

use crate::distributions::gamma::SimdGamma;
use crate::distributions::normal::SimdNormal;
use crate::simd_rng::Deterministic;
use crate::simd_rng::SeedExt;
use crate::simd_rng::Unseeded;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct VG<T: FloatExt, S: SeedExt = Unseeded> {
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

impl<T: FloatExt> VG<T> {
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

impl<T: FloatExt> VG<T, Deterministic> {
  pub fn seeded(mu: T, sigma: T, nu: T, n: usize, x0: Option<T>, t: Option<T>, seed: u64) -> Self {
    assert!(nu > T::zero(), "nu must be positive");
    Self {
      mu,
      sigma,
      nu,
      n,
      x0,
      t,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> VG<T, S> {
  #[inline]
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for VG<T, S> {
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
    let gamma = SimdGamma::new(dt / self.nu, self.nu);
    let mut seed = self.seed;
    let mut rng = seed.rng();
    let gammas = Array1::random_using(self.n - 1, &gamma, &mut rng);
    let mut z = Array1::<T>::zeros(self.n - 1);
    let z_slice = z.as_slice_mut().expect("VG normals must be contiguous");
    let normal = SimdNormal::<T>::from_seed_source(T::zero(), T::one(), &mut seed);
    normal.fill_slice_fast(z_slice);

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
    let p = VG::new(0.1_f64, 0.2, 0.3, 1, Some(2.5), Some(1.0));
    let x = p.sample();
    assert_eq!(x.len(), 1);
    assert_eq!(x[0], 2.5);
  }
}

py_process_1d!(PyVG, VG,
  sig: (mu, sigma, nu, n, x0=None, t=None, dtype=None),
  params: (mu: f64, sigma: f64, nu: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
