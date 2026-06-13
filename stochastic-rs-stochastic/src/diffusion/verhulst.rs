//! # Verhulst
//!
//! $$
//! dX_t=rX_t\left(1-\frac{X_t}{K}\right)dt+\sigma X_t dW_t
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

/// Verhulst (logistic) diffusion
/// dX_t = r X_t (1 - X_t / K) dt + sigma X_t dW_t
pub struct Verhulst<T: FloatExt, S: SeedExt = Unseeded> {
  /// Risk-free rate / drift adjustment parameter.
  pub r: T,
  /// Jump-size adjustment / shape parameter.
  pub k: T,
  /// Diffusion / noise scale parameter.
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// If true, clamp the state into [0, K] each step
  pub clamp: Option<bool>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Verhulst<T, S> {
  pub fn new(
    r: T,
    k: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    clamp: Option<bool>,
    seed: S,
  ) -> Self {
    Self {
      r,
      k,
      sigma,
      n,
      x0,
      t,
      clamp,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Verhulst<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = VerhulstSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> VerhulstSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    VerhulstSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      r: self.r,
      k: self.k,
      diff_scale: self.sigma,
      clamp: self.clamp.unwrap_or(true),
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`Verhulst`] sampling state: precomputed Euler scales and the owned
/// Gaussian source.
#[doc(hidden)]
pub struct VerhulstSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  r: T,
  k: T,
  diff_scale: T,
  clamp: bool,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> VerhulstSampler<T> {
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
      let xi = prev;
      let drift = self.r * xi * (T::one() - xi / self.k) * self.dt;
      let diff = self.diff_scale * xi * *z;
      let mut next = xi + drift + diff;
      if self.clamp {
        next = next.clamp(T::zero(), self.k);
      }
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for VerhulstSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("Verhulst output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyVerhulst, Verhulst,
  sig: (r, k, sigma, n, x0=None, t=None, clamp=None, seed=None, dtype=None),
  params: (r: f64, k: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, clamp: Option<bool>)
);
