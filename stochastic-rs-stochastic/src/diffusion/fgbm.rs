//! # fGBM
//!
//! $$
//! dS_t=\mu S_t\,dt+\sigma S_t\,dB_t^H
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Backend;
use crate::device::Cpu;
use crate::noise::fgn::Fgn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct Fgbm<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
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
  fgn: Fgn<T, Unseeded, B>,
}

impl<T: FloatExt, S: SeedExt> Fgbm<T, S, Cpu> {
  #[must_use]
  pub fn new(hurst: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    assert!(n >= 2, "n must be at least 2");

    Self {
      hurst,
      mu,
      sigma,
      n,
      x0,
      t,
      seed,
      fgn: Fgn::new(hurst, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt, B: Backend> ProcessExt<T> for Fgbm<T, S, B> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.fgn.dt();
    let fgn = self.fgn.noise(&self.seed.derive());

    let mut fgbm = Array1::<T>::zeros(self.n);
    fgbm[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      fgbm[i] = fgbm[i - 1] + self.mu * fgbm[i - 1] * dt + self.sigma * fgbm[i - 1] * fgn[i - 1];
    }

    fgbm
  }
}

impl<T: FloatExt, S: SeedExt, B> Fgbm<T, S, B> {
  backend_switch_on!(Fgbm<T, S> { hurst, mu, sigma, n, x0, t, seed }, fgn);
}

py_process_1d!(PyFgbm, Fgbm,
  sig: (hurst, mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (hurst: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::Fgbm;
  use crate::device::Cpu;
  use crate::traits::ProcessExt;

  #[test]
  fn fgbm_on_cpu_matches_plain_sample() {
    let mk = || Fgbm::<f64, _>::new(0.7, 0.1, 0.2, 256, Some(1.0), Some(1.0), Deterministic::new(7));
    let plain = mk().sample();
    let on_cpu = mk().on::<Cpu>().sample();

    assert_eq!(plain.len(), on_cpu.len());
    for (a, b) in plain.iter().zip(on_cpu.iter()) {
      assert_eq!(a, b);
    }
  }
}
