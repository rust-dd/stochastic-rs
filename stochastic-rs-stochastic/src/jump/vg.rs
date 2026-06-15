//! # Vg
//!
//! $$
//! X_t=\theta G_t+\sigma W_{G_t},\quad G_t\sim\Gamma(\nu^{-1}t,\nu)
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
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

impl<T: FloatExt, S: SeedExt> Vg<T, S> {
  pub fn new(mu: T, sigma: T, nu: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    assert!(nu > T::zero(), "nu must be positive");
    Self {
      mu,
      sigma,
      nu,
      n,
      x0,
      t,
      seed,
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
  type Sampler<'s>
    = VgSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> VgSampler<T> {
    // Gamma subordinator and standard-normal source derived from `self.seed`
    // in the same order as the legacy `sample()`, so the first fill matches
    // bit-for-bit; both owned sources advance on reuse for independent paths.
    let dt = self.dt();
    VgSampler {
      n: self.n,
      mu: self.mu,
      sigma: self.sigma,
      x0: self.x0.unwrap_or(T::zero()),
      gamma: SimdGamma::<T>::new(dt / self.nu, self.nu, &self.seed),
      normal: SimdNormal::<T>::new(T::zero(), T::one(), &self.seed),
    }
  }
}

/// Reusable [`Vg`] sampling state: owns the gamma subordinator and the
/// Gaussian source so a Monte-Carlo loop pays their setup once.
#[doc(hidden)]
pub struct VgSampler<T: FloatExt> {
  n: usize,
  mu: T,
  sigma: T,
  x0: T,
  gamma: SimdGamma<T>,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> VgSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }

    let mut gammas = Array1::<T>::zeros(out.len() - 1);
    self.gamma.fill_slice_fast(gammas.as_slice_mut().unwrap());
    let mut z = Array1::<T>::zeros(out.len() - 1);
    self.normal.fill_slice_fast(z.as_slice_mut().unwrap());

    for i in 1..out.len() {
      out[i] = out[i - 1] + self.mu * gammas[i - 1] + self.sigma * gammas[i - 1].sqrt() * z[i - 1];
    }
  }
}

impl<T: FloatExt> PathSampler<T> for VgSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(out.as_slice_mut().expect("Vg output must be contiguous"));
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::ProcessExt;

  #[test]
  fn n_eq_1_keeps_initial_value() {
    let p = Vg::new(0.1_f64, 0.2, 0.3, 1, Some(2.5), Some(1.0), Unseeded);
    let x = p.sample();
    assert_eq!(x.len(), 1);
    assert_eq!(x[0], 2.5);
  }
}

py_process_1d!(PyVg, Vg,
  sig: (mu, sigma, nu, n, x0=None, t=None, seed=None, dtype=None),
  params: (mu: f64, sigma: f64, nu: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
