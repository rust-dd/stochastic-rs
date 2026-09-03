//! # ThreeHalf
//!
//! $$
//! dX_t=\kappa X_t(\mu-X_t)\,dt+\sigma X_t^{3/2}\,dW_t
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

pub struct ThreeHalf<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Mean-reversion speed parameter.
  pub kappa: T,
  /// Long-run mean level μ the drift `κX_t(μ−X_t)` reverts toward.
  pub mu: T,
  /// Diffusion scale σ multiplying `X_t^{3/2} dW_t`.
  pub sigma: T,
  /// Number of points sampled along the 3/2 path.
  pub n: usize,
  /// Initial value X₀ of the 3/2 path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// Sampling backend marker (compile-time): [`Cpu`] by default, a device
  /// marker after [`on`](Self::on). Public so `..Default::default()` struct
  /// updates keep working; it carries no data.
  pub backend: PhantomData<B>,
}

impl<T: FloatExt, S: SeedExt> ThreeHalf<T, S> {
  pub fn new(kappa: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    Self {
      backend: PhantomData,
      kappa,
      mu,
      sigma,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> ThreeHalf<T, S, B> {}

backend_switch!([T: FloatExt, S: SeedExt] ThreeHalf<T, S> { kappa, mu, sigma, n, x0, t, seed } via host);

impl<T: FloatExt, S: SeedExt, B: HostBackend> ProcessExt<T> for ThreeHalf<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = ThreeHalfSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> ThreeHalfSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    ThreeHalfSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      kappa: self.kappa,
      mu: self.mu,
      sigma: self.sigma,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`ThreeHalf`] sampling state: precomputed Euler step and the owned
/// Gaussian source.
#[doc(hidden)]
pub struct ThreeHalfSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  kappa: T,
  mu: T,
  sigma: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> ThreeHalfSampler<T> {
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
      let next = prev
        + self.kappa * prev * (self.mu - prev) * self.dt
        + self.sigma * prev.abs().powf(T::from_f64_fast(1.5)) * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for ThreeHalfSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("ThreeHalf output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyThreeHalf, ThreeHalf,
  sig: (kappa, mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (kappa: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
