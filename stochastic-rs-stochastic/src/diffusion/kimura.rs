//! # Kimura
//!
//! $$
//! dX_t=(a+bX_t)dt+\sigma\sqrt{X_t}\,dW_t
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

/// Kimura / Wright–Fisher diffusion
/// dX_t = a X_t (1 - X_t) dt + sigma sqrt(X_t (1 - X_t)) dW_t
pub struct Kimura<T: FloatExt, S: SeedExt = Unseeded> {
  /// Model coefficient / user-supplied drift term.
  pub a: T,
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
}

impl<T: FloatExt, S: SeedExt> Kimura<T, S> {
  pub fn new(a: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    Self {
      a,
      sigma,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Kimura<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = KimuraSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> KimuraSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    KimuraSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      a: self.a,
      diff_scale: self.sigma,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`Kimura`] sampling state.
#[doc(hidden)]
pub struct KimuraSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  a: T,
  diff_scale: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> KimuraSampler<T> {
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
      // enforce [0,1] domain when computing coefficients
      let xi = prev.clamp(T::zero(), T::one());
      let sqrt_term = (xi * (T::one() - xi)).sqrt();
      let drift = self.a * xi * (T::one() - xi) * self.dt;
      let diff = self.diff_scale * sqrt_term * *z;
      let mut next = xi + drift + diff;
      next = next.clamp(T::zero(), T::one());
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for KimuraSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("Kimura output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyKimura, Kimura,
  sig: (a, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (a: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
