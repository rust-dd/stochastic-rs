//! # Quadratic
//!
//! $$
//! dX_t=(\gamma X_t^2+\beta X_t+\alpha)dt+\sigma X_t\,dW_t
//! $$
//!

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

/// Quadratic diffusion
/// dX_t = (alpha + beta X_t + gamma X_t^2) dt + sigma X_t dW_t
pub struct Quadratic<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Constant term α in the drift `α + βX_t + γX_t²`.
  pub alpha: T,
  /// Linear coefficient β in the drift.
  pub beta: T,
  /// Quadratic coefficient γ in the drift (adds curvature beyond the
  /// linear OU-style term).
  pub gamma: T,
  /// Proportional diffusion scale σ multiplying `X_t dW_t`.
  pub sigma: T,
  /// Number of points sampled along the quadratic-diffusion path.
  pub n: usize,
  /// Initial value X₀ of the quadratic-diffusion path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on) or [`on_device`](Self::on_device).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Quadratic<T, S> {
  pub fn new(
    alpha: T,
    beta: T,
    gamma: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self {
      backend: Cpu,
      alpha,
      beta,
      gamma,
      sigma,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Quadratic<T, S, B> {}

backend_switch!([T: FloatExt, S: SeedExt] Quadratic<T, S> { alpha, beta, gamma, sigma, n, x0, t, seed } via host);

impl<T: FloatExt, S: SeedExt, B: HostBackend> ProcessExt<T> for Quadratic<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = QuadraticSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> QuadraticSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    QuadraticSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      alpha: self.alpha,
      beta: self.beta,
      gamma: self.gamma,
      diff_scale: self.sigma,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`Quadratic`] sampling state: precomputed Euler scales and the
/// owned Gaussian source.
#[doc(hidden)]
pub struct QuadraticSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  alpha: T,
  beta: T,
  gamma: T,
  diff_scale: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> QuadraticSampler<T> {
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
      let xi = prev;
      let drift = (self.alpha + self.beta * xi + self.gamma * xi * xi) * self.dt;
      let next = xi + drift + self.diff_scale * xi * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for QuadraticSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("Quadratic output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyQuadratic, Quadratic,
  sig: (alpha, beta, gamma, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (alpha: f64, beta: f64, gamma: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
