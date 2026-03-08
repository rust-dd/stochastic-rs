//! # fGBM
//!
//! $$
//! dS_t=\mu S_t\,dt+\sigma S_t\,dB_t^H
//! $$
//!
use ndarray::Array1;

use crate::simd_rng::Deterministic;
use crate::simd_rng::SeedExt;
use crate::simd_rng::Unseeded;
use crate::stochastic::noise::fgn::FGN;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct FGBM<T: FloatExt, S: SeedExt = Unseeded> {
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
  fgn: FGN<T>,
}

impl<T: FloatExt> FGBM<T> {
  #[must_use]
  pub fn new(hurst: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    assert!(n >= 2, "n must be at least 2");

    Self {
      hurst,
      mu,
      sigma,
      n,
      x0,
      t,
      seed: Unseeded,
      fgn: FGN::new(hurst, n - 1, t),
    }
  }
}

impl<T: FloatExt> FGBM<T, Deterministic> {
  #[must_use]
  pub fn seeded(hurst: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>, seed: u64) -> Self {
    assert!(n >= 2, "n must be at least 2");

    Self {
      hurst,
      mu,
      sigma,
      n,
      x0,
      t,
      seed: Deterministic(seed),
      fgn: FGN::new(hurst, n - 1, t),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for FGBM<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut seed = self.seed;
    let dt = self.fgn.dt();
    let fgn = self.fgn.sample_cpu_impl(seed.derive());

    let mut fgbm = Array1::<T>::zeros(self.n);
    fgbm[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      fgbm[i] = fgbm[i - 1] + self.mu * fgbm[i - 1] * dt + self.sigma * fgbm[i - 1] * fgn[i - 1];
    }

    fgbm
  }
}

py_process_1d!(PyFGBM, FGBM,
  sig: (hurst, mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (hurst: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
