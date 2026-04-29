//! # Fvasicek
//!
//! $$
//! dr_t=a(b-r_t)dt+\sigma dB_t^H
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::diffusion::fou::Fou;
use crate::traits::FloatExt;
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

impl<T: FloatExt> FVasicek<T> {
  pub fn new(hurst: T, theta: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      seed: Unseeded,
      fou: Fou::new(hurst, theta, mu, sigma, n, x0, t),
    }
  }
}

impl<T: FloatExt> FVasicek<T, Deterministic> {
  pub fn seeded(
    hurst: T,
    theta: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: u64,
  ) -> Self {
    let s = Deterministic::new(seed);
    let child = s.derive();
    Self {
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      seed: Deterministic::new(seed),
      fou: Fou::seeded(hurst, theta, mu, sigma, n, x0, t, child.current()),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for FVasicek<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Array1<T> {
    self.fou.sample()
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
    let v = FVasicek::<f64>::new(0.7, 0.5, 0.04, 0.01, 64, Some(0.05), Some(1.0));
    let path = v.sample();
    assert_eq!(path.len(), 64);
  }
}
