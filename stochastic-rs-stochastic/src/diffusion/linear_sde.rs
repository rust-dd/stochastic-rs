//! # LinearSDE
//!
//! $$
//! dX_t=(a+bX_t)\,dt+cX_t\,dW_t
//! $$
//!
use std::marker::PhantomData;

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::device::HostBackend;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Clone, Copy)]
pub struct LinearSDE<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Drift intercept a in `(a + bX_t)dt`.
  pub a: T,
  /// Drift slope b (linear mean-reversion-like coefficient) in
  /// `(a + bX_t)dt`.
  pub b: T,
  /// Proportional diffusion scale c multiplying `X_t dW_t`.
  pub c: T,
  /// Number of points sampled along the linear-SDE path.
  pub n: usize,
  /// Initial value X₀ of the linear-SDE path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub seed: S,
  /// Sampling backend marker (compile-time): [`Cpu`] by default, a device
  /// marker after [`on`](Self::on). Public so `..Default::default()` struct
  /// updates keep working; it carries no data.
  pub backend: PhantomData<B>,
}

impl<T: FloatExt, S: SeedExt> LinearSDE<T, S> {
  pub fn new(a: T, b: T, c: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    Self {
      backend: PhantomData,
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

impl<T: FloatExt, S: SeedExt, B> LinearSDE<T, S, B> {}

backend_switch!([T: FloatExt, S: SeedExt] LinearSDE<T, S> { a, b, c, n, x0, t, seed } via host);

impl<T: FloatExt, S: SeedExt, B: HostBackend> ProcessExt<T> for LinearSDE<T, S, B> {
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
    self.normal.fill_slice(tail);
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
