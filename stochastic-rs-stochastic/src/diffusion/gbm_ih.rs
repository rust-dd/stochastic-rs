//! # Gbm Ih
//!
//! $$
//! dS_t=\mu(t)S_t\,dt+\sigma(t)S_t\,dW_t
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

/// Inhomogeneous Gbm with time-dependent volatility
/// dX_t = mu X_t dt + sigma(t) X_t dW_t
pub struct GbmIh<T: FloatExt, S: SeedExt = Unseeded> {
  /// Drift / long-run mean-level parameter.
  pub mu: T,
  /// Baseline sigma used when `sigmas` is None
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Optional per-step volatilities (length must be n-1)
  pub sigmas: Option<Array1<T>>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> GbmIh<T, S> {
  /// Create a new GbmIh instance with the given parameters.
  pub fn new(
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    sigmas: Option<Array1<T>>,
    seed: S,
  ) -> Self {
    if let Some(s) = &sigmas {
      assert_eq!(s.len(), n.saturating_sub(1), "sigmas length must be n - 1");
    }

    Self {
      mu,
      sigma,
      n,
      x0,
      t,
      sigmas,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for GbmIh<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = GbmIhSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> GbmIhSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    GbmIhSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      drift_scale: self.mu * dt,
      sigma: self.sigma,
      sigmas: self.sigmas.clone(),
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`GbmIh`] sampling state.
#[doc(hidden)]
pub struct GbmIhSampler<T: FloatExt> {
  n: usize,
  x0: T,
  drift_scale: T,
  sigma: T,
  sigmas: Option<Array1<T>>,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> GbmIhSampler<T> {
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
    for (i, z) in tail.iter_mut().enumerate() {
      let sigma_i = self.sigmas.as_ref().map(|s| s[i]).unwrap_or(self.sigma);
      let next = prev + self.drift_scale * prev + sigma_i * prev * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for GbmIhSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("GbmIh output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyGbmIh, GbmIh,
  sig: (mu, sigma, n, x0=None, t=None, sigmas=None, seed=None, dtype=None),
  params: (mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, sigmas: Option<Vec<f64>>)
);
