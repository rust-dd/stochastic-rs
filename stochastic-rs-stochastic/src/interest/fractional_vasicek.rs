//! # Fvasicek
//!
//! $$
//! dr_t=a(b-r_t)dt+\sigma dB_t^H
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::diffusion::fou::Fou;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

pub struct FVasicek<T: FloatExt, S: SeedExt = Unseeded> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
  /// Long-run target level / model location parameter.
  pub theta: T,
  /// Drift / long-run mean-level parameter.
  pub mu: T,
  /// Diffusion / noise scale parameter.
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
  /// Model parameter controlling process dynamics.
  pub fou: Fou<T, S>,
}

impl<T: FloatExt, S: SeedExt> FVasicek<T, S> {
  pub fn new(
    hurst: T,
    theta: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self {
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      fou: Fou::new(hurst, theta, mu, sigma, n, x0, t, seed.derive()),
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for FVasicek<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = FVasicekSampler<'s, T, S>
  where
    Self: 's;

  fn sampler(&self) -> FVasicekSampler<'_, T, S> {
    FVasicekSampler { proc: self }
  }
}

/// Reusable [`FVasicek`] sampling state. Borrows the process so each call
/// resamples the wrapped fractional OU through its `Arc`-shared fGn FFT plan.
///
/// The inner [`Fou`] owns no persistent Gaussian source, so this samples
/// through [`Fou::sample`] each call: the FFT plan and circulant eigenvalues
/// are `Arc`-shared and reused, only the per-call `SimdNormal` is rebuilt.
#[doc(hidden)]
pub struct FVasicekSampler<'a, T: FloatExt, S: SeedExt> {
  proc: &'a FVasicek<T, S>,
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for FVasicekSampler<'_, T, S> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    *out = self.proc.fou.sample();
  }

  fn sample(&mut self) -> Array1<T> {
    self.proc.fou.sample()
  }
}

py_process_1d!(PyFVasicek, FVasicek,
  sig: (hurst, theta, mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (hurst: f64, theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn sample_length_matches_n() {
    let v = FVasicek::<f64>::new(0.7, 0.5, 0.04, 0.01, 64, Some(0.05), Some(1.0), Unseeded);
    let path = v.sample();
    assert_eq!(path.len(), 64);
  }
}
