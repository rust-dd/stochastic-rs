//! # Fjacobi
//!
//! $$
//! dX_t=\kappa(\theta-X_t)dt+\sigma\sqrt{X_t(1-X_t)}\,dB_t^H
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::buffer::array1_from_fill;
use crate::device::Backend;
use crate::device::Cpu;
use crate::noise::fgn::Fgn;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

pub struct FJacobi<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
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
  fgn: Fgn<T, Unseeded, B>,
}

impl<T: FloatExt, S: SeedExt> FJacobi<T, S, Cpu> {
  #[must_use]
  pub fn new(
    hurst: T,
    alpha: T,
    beta: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
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
      seed,
      fgn: Fgn::new(hurst, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt, B: Backend> ProcessExt<T> for FJacobi<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = FJacobiSampler<'s, T, S, B>
  where
    Self: 's;

  /// A CPU sampler borrowing the process for its inner [`Fgn`] (`Arc`-shared
  /// FFT plan + eigenvalues) and seed source. The first `sample` derives the
  /// same child seed the legacy `sample()` did — bit-identical — and each
  /// subsequent call advances the seed for an independent path.
  fn sampler(&self) -> FJacobiSampler<'_, T, S, B> {
    FJacobiSampler { fjacobi: self }
  }
}

/// Reusable [`FJacobi`] sampling state: borrows the process for its inner
/// [`Fgn`] and seed source. The path is an Euler discretisation of
/// `dX = (alpha - beta X) dt + sigma sqrt(X(1 - X)) dB^H`, clamped into `[0, 1]`.
#[doc(hidden)]
pub struct FJacobiSampler<'a, T: FloatExt, S: SeedExt, B> {
  fjacobi: &'a FJacobi<T, S, B>,
}

impl<T: FloatExt, S: SeedExt, B: Backend> FJacobiSampler<'_, T, S, B> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    let p = self.fjacobi;
    let dt = p.fgn.dt();
    let fgn = p.fgn.noise(&p.seed.derive());

    out[0] = p.x0.unwrap_or(T::zero());
    let mut prev = out[0];
    for (dst, inc) in out[1..].iter_mut().zip(fgn.iter()) {
      let next = match prev {
        _ if prev <= T::zero() => T::zero(),
        _ if prev >= T::one() => T::one(),
        _ => {
          prev + (p.alpha - p.beta * prev) * dt + p.sigma * (prev * (T::one() - prev)).sqrt() * *inc
        }
      };
      *dst = next;
      prev = next;
    }
  }
}

impl<T: FloatExt, S: SeedExt, B: Backend> PathSampler<T> for FJacobiSampler<'_, T, S, B> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("FJacobi output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.fjacobi.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

backend_switch!([T: FloatExt, S: SeedExt] FJacobi<T, S> { hurst, alpha, beta, sigma, n, x0, t, seed } via fgn);

py_process_1d!(PyFJacobi, FJacobi,
  sig: (hurst, alpha, beta, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (hurst: f64, alpha: f64, beta: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
