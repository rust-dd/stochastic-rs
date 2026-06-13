//! # LinearSDE
//!
//! $$
//! dX_t=(a+bX_t)\,dt+cX_t\,dW_t
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
pub struct LinearSDE<T: FloatExt, S: SeedExt = Unseeded> {
  pub a: T,
  pub b: T,
  pub c: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> LinearSDE<T, S> {
  pub fn new(a: T, b: T, c: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    Self {
      a,
      b,
      c,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for LinearSDE<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = LinearSDESampler<T>
  where
    Self: 's;

  fn sampler(&self) -> LinearSDESampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    LinearSDESampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      a: self.a,
      b: self.b,
      c: self.c,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`LinearSDE`] sampling state.
#[doc(hidden)]
pub struct LinearSDESampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  a: T,
  b: T,
  c: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> LinearSDESampler<T> {
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
      let next = prev + (self.a + self.b * prev) * self.dt + self.c * prev * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for LinearSDESampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("LinearSDE output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyLinearSDE, LinearSDE,
  sig: (a, b, c, n, x0=None, t=None, seed=None, dtype=None),
  params: (a: f64, b: f64, c: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
