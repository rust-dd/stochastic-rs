//! # Logistic
//!
//! $$
//! dX_t=X_t(1-aX_t)\,dt+bX_t\,dW_t
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

pub struct Logistic<T: FloatExt, S: SeedExt = Unseeded> {
  /// Carrying-capacity parameter.
  pub a: T,
  /// Diffusion / noise scale parameter.
  pub b: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Logistic<T, S> {
  pub fn new(a: T, b: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    Self {
      a,
      b,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Logistic<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = LogisticSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> LogisticSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    LogisticSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      a: self.a,
      b: self.b,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`Logistic`] sampling state.
#[doc(hidden)]
pub struct LogisticSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  a: T,
  b: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> LogisticSampler<T> {
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
      let next = prev + prev * (T::one() - self.a * prev) * self.dt + self.b * prev * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for LogisticSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("Logistic output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyLogistic, Logistic,
  sig: (a, b, n, x0=None, t=None, seed=None, dtype=None),
  params: (a: f64, b: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
