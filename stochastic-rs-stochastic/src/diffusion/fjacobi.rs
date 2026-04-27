//! # Fjacobi
//!
//! $$
//! dX_t=\kappa(\theta-X_t)dt+\sigma\sqrt{X_t(1-X_t)}\,dB_t^H
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::noise::fgn::Fgn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct FJacobi<T: FloatExt, S: SeedExt = Unseeded> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
  /// Model shape / loading parameter.
  pub alpha: T,
  /// Model slope / loading parameter.
  pub beta: T,
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
  fgn: Fgn<T>,
}

impl<T: FloatExt> FJacobi<T> {
  #[must_use]
  pub fn new(hurst: T, alpha: T, beta: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    assert!(n >= 2, "n must be at least 2");
    assert!(alpha > T::zero(), "alpha must be positive");
    assert!(beta > T::zero(), "beta must be positive");
    assert!(sigma > T::zero(), "sigma must be positive");
    assert!(alpha < beta, "alpha must be less than beta");

    Self {
      hurst,
      alpha,
      beta,
      sigma,
      n,
      x0,
      t,
      seed: Unseeded,
      fgn: Fgn::new(hurst, n - 1, t),
    }
  }
}

impl<T: FloatExt> FJacobi<T, Deterministic> {
  #[must_use]
  pub fn seeded(
    hurst: T,
    alpha: T,
    beta: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: u64,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");
    assert!(alpha > T::zero(), "alpha must be positive");
    assert!(beta > T::zero(), "beta must be positive");
    assert!(sigma > T::zero(), "sigma must be positive");
    assert!(alpha < beta, "alpha must be less than beta");

    Self {
      hurst,
      alpha,
      beta,
      sigma,
      n,
      x0,
      t,
      seed: Deterministic(seed),
      fgn: Fgn::new(hurst, n - 1, t),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for FJacobi<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut seed = self.seed;
    let dt = self.fgn.dt();
    let fgn = self.fgn.sample_cpu_impl(seed.derive());

    let mut fjacobi = Array1::<T>::zeros(self.n);
    fjacobi[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      fjacobi[i] = match fjacobi[i - 1] {
        _ if fjacobi[i - 1] <= T::zero() && i > 0 => T::zero(),
        _ if fjacobi[i - 1] >= T::one() && i > 0 => T::one(),
        _ => {
          fjacobi[i - 1]
            + (self.alpha - self.beta * fjacobi[i - 1]) * dt
            + self.sigma * (fjacobi[i - 1] * (T::one() - fjacobi[i - 1])).sqrt() * fgn[i - 1]
        }
      };
    }

    fjacobi
  }
}

py_process_1d!(PyFJacobi, FJacobi,
  sig: (hurst, alpha, beta, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (hurst: f64, alpha: f64, beta: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
