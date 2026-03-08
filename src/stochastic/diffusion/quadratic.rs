//! # Quadratic
//!
//! $$
//! dX_t=(aX_t^2+bX_t+c)dt+\sigma X_t dW_t
//! $$
//!
use ndarray::Array1;
use ndarray::s;

use crate::distributions::normal::SimdNormal;
use crate::simd_rng::Deterministic;
use crate::simd_rng::SeedExt;
use crate::simd_rng::Unseeded;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Quadratic diffusion
/// dX_t = (alpha + beta X_t + gamma X_t^2) dt + sigma X_t dW_t
pub struct Quadratic<T: FloatExt, S: SeedExt = Unseeded> {
  /// Model shape / loading parameter.
  pub alpha: T,
  /// Model slope / loading parameter.
  pub beta: T,
  /// Model asymmetry / nonlinearity parameter.
  pub gamma: T,
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

impl<T: FloatExt> Quadratic<T> {
  pub fn new(alpha: T, beta: T, gamma: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      alpha,
      beta,
      gamma,
      sigma,
      n,
      x0,
      t,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> Quadratic<T, Deterministic> {
  pub fn seeded(alpha: T, beta: T, gamma: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>, seed: u64) -> Self {
    Self {
      alpha,
      beta,
      gamma,
      sigma,
      n,
      x0,
      t,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Quadratic<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut x = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return x;
    }

    x[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return x;
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let sqrt_dt = dt.sqrt();
    let diff_scale = self.sigma;
    let mut prev = x[0];
    let mut tail_view = x.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("Quadratic output tail must be contiguous");
    let mut seed = self.seed;
    let normal = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &mut seed);
    normal.fill_slice_fast(tail);

    for z in tail.iter_mut() {
      let xi = prev;
      let drift = (self.alpha + self.beta * xi + self.gamma * xi * xi) * dt;
      let next = xi + drift + diff_scale * xi * *z;
      *z = next;
      prev = next;
    }

    x
  }
}

py_process_1d!(PyQuadratic, Quadratic,
  sig: (alpha, beta, gamma, sigma, n, x0=None, t=None, dtype=None),
  params: (alpha: f64, beta: f64, gamma: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
