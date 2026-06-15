//! # Ou
//!
//! $$
//! dX_t=\kappa(\theta-X_t)\,dt+\sigma\,dW_t
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Clone, Copy)]
pub struct Ou<T: FloatExt, S: SeedExt = Unseeded> {
  /// Mean-reversion speed (κ in the SDE `dX = κ(θ − X) dt + σ dW`). Controls
  /// how fast `X` is pulled back toward [`mu`](Self::mu).
  pub theta: T,
  /// Long-run mean level (θ in the SDE). The value `X` reverts to as
  /// `t → ∞`.
  pub mu: T,
  /// Diffusion / noise scale parameter (σ in the SDE).
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Ou<T, S> {
  pub fn new(theta: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    Self {
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Ou<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = OuSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> OuSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    OuSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      mu: self.mu,
      drift_scale: self.theta * dt,
      diff_scale: self.sigma,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`Ou`] sampling state: precomputed mean-reversion scales and the
/// owned Gaussian source.
#[doc(hidden)]
pub struct OuSampler<T: FloatExt> {
  n: usize,
  x0: T,
  mu: T,
  drift_scale: T,
  diff_scale: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> OuSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let tail = &mut out[1..];
    self.normal.fill_slice_fast(tail);
    let mut prev = self.x0;
    for z in tail.iter_mut() {
      let next = prev + self.drift_scale * (self.mu - prev) + self.diff_scale * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for OuSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Ou output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyOu, Ou,
  sig: (theta, mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
