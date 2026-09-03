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
use crate::device::Cpu;
use crate::device::FgnBackend;
use crate::noise::fgn::Fgn;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Clone)]
pub struct FJacobi<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
  /// Linear-drift intercept (κθ combined) in the reparametrized drift
  /// `alpha - beta·X`, equivalent to `κ(θ-X)` with `alpha = κθ`. Must be
  /// less than `beta` so the implied θ = alpha/beta stays in (0, 1), the
  /// Jacobi boundary requirement.
  pub alpha: T,
  /// Linear-drift slope (mean-reversion speed κ) in `alpha - beta·X`.
  pub beta: T,
  /// Diffusion scale σ multiplying `√(X_t(1-X_t)) dB_t^H`.
  pub sigma: T,
  /// Number of points sampled along the fractional Jacobi path.
  pub n: usize,
  /// Initial value X₀ of the fractional Jacobi path (clamped into [0, 1]).
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
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

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> ProcessExt<T> for FJacobi<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = FJacobiSampler<'s, T, S, B>
  where
    Self: 's;

  /// A CPU sampler borrowing the process for its inner [`Fgn`] (`Arc`-shared
  /// FFT plan + eigenvalues) and owning a seed derived once at construction.
  /// Deriving (not cloning) is what decorrelates chunks: the derived value
  /// is `self.seed`'s *mixed* next tick, not a raw snapshot, so chunk `i`'s
  /// basis and chunk `i+1`'s basis are hash-scrambled relative to each
  /// other rather than one raw stride apart. `fill_path` then uses this
  /// owned seed *directly* (no further derive) — exactly one derive from
  /// `self.seed` per chunk, matching what the legacy per-call `derive()`
  /// consumed, so the first path reproduces the legacy stream bit-for-bit.
  /// Repeat calls on one sampler advance the same owned seed further, for
  /// an independent path each time.
  fn sampler(&self) -> FJacobiSampler<'_, T, S, B> {
    FJacobiSampler {
      fjacobi: self,
      seed: self.seed.derive(),
    }
  }
}

/// Reusable [`FJacobi`] sampling state: borrows the process for its inner
/// [`Fgn`] and owns a seed derived once at construction. The path is an
/// Euler discretisation of `dX = (alpha - beta X) dt + sigma sqrt(X(1 - X))
/// dB^H`, clamped into `[0, 1]`.
#[doc(hidden)]
pub struct FJacobiSampler<'a, T: FloatExt, S: SeedExt, B> {
  fjacobi: &'a FJacobi<T, S, B>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> FJacobiSampler<'_, T, S, B> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    let p = self.fjacobi;
    let dt = p.fgn.dt();
    let fgn = p.fgn.noise(&self.seed);

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

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> PathSampler<T> for FJacobiSampler<'_, T, S, B> {
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
  params: (hurst: f64, alpha: f64, beta: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);
