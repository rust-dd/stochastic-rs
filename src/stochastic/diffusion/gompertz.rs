//! # Gompertz
//!
//! $$
//! dX_t=aX_t\ln\!\left(\frac{K}{X_t}\right)dt+\sigma X_t dW_t
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

/// Gompertz diffusion
/// dX_t = (a - b ln X_t) X_t dt + sigma X_t dW_t
pub struct Gompertz<T: FloatExt, S: SeedExt = Unseeded> {
  /// Model coefficient / user-supplied drift term.
  pub a: T,
  /// Model coefficient / user-supplied diffusion term.
  pub b: T,
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

impl<T: FloatExt> Gompertz<T> {
  pub fn new(a: T, b: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      a,
      b,
      sigma,
      n,
      x0,
      t,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> Gompertz<T, Deterministic> {
  pub fn seeded(a: T, b: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>, seed: u64) -> Self {
    Self {
      a,
      b,
      sigma,
      n,
      x0,
      t,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Gompertz<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut x = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return x;
    }

    let threshold = T::from_f64_fast(1e-12);
    x[0] = self.x0.unwrap_or(T::zero()).max(threshold);
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
      .expect("Gompertz output tail must be contiguous");
    let mut seed = self.seed;
    let normal = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &mut seed);
    normal.fill_slice_fast(tail);

    for z in tail.iter_mut() {
      let xi = prev.max(threshold);
      let drift = (self.a - self.b * xi.ln()) * xi * dt;
      let diff = diff_scale * xi * *z;
      let next = xi + drift + diff;
      let clamped = next.max(threshold);
      *z = clamped;
      prev = clamped;
    }

    x
  }
}

py_process_1d!(PyGompertz, Gompertz,
  sig: (a, b, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (a: f64, b: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
