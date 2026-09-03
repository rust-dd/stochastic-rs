//! # Pearson
//!
//! $$
//! dX_t=\kappa(\mu-X_t)\,dt+\sqrt{2\kappa(aX_t^2+bX_t+c)}\,dW_t
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
pub struct Pearson<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Mean-reversion speed κ.
  pub kappa: T,
  /// Long-run mean level μ.
  pub mu: T,
  /// Quadratic coefficient a inside the diffusion bracket
  /// `2κ(aX_t²+bX_t+c)`.
  pub a: T,
  /// Linear coefficient b inside the diffusion bracket.
  pub b: T,
  /// Constant coefficient c inside the diffusion bracket.
  pub c: T,
  /// Number of points sampled along the Pearson-diffusion path.
  pub n: usize,
  /// Initial value X₀ of the Pearson-diffusion path.
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

impl<T: FloatExt, S: SeedExt> Pearson<T, S> {
  pub fn new(
    kappa: T,
    mu: T,
    a: T,
    b: T,
    c: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self {
      backend: PhantomData,
      kappa,
      mu,
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

impl<T: FloatExt, S: SeedExt, B> Pearson<T, S, B> {}

backend_switch!([T: FloatExt, S: SeedExt] Pearson<T, S> { kappa, mu, a, b, c, n, x0, t, seed } via host);

impl<T: FloatExt, S: SeedExt, B: HostBackend> ProcessExt<T> for Pearson<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = PearsonSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> PearsonSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    PearsonSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      kappa: self.kappa,
      mu: self.mu,
      a: self.a,
      b: self.b,
      c: self.c,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`Pearson`] sampling state: precomputed Euler step and the owned
/// Gaussian source.
#[doc(hidden)]
pub struct PearsonSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  kappa: T,
  mu: T,
  a: T,
  b: T,
  c: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> PearsonSampler<T> {
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
      let diff_inner =
        T::from_f64_fast(2.0) * self.kappa * (self.a * prev * prev + self.b * prev + self.c);
      let next = prev + self.kappa * (self.mu - prev) * self.dt + diff_inner.abs().sqrt() * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for PearsonSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("Pearson output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyPearson, Pearson,
  sig: (kappa, mu, a, b, c, n, x0=None, t=None, seed=None, dtype=None),
  params: (kappa: f64, mu: f64, a: f64, b: f64, c: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
