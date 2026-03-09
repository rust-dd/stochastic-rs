//! # Logistic
//!
//! $$
//! dX_t=X_t(1-aX_t)\,dt+bX_t\,dW_t
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

impl<T: FloatExt> Logistic<T> {
  pub fn new(a: T, b: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      a,
      b,
      n,
      x0,
      t,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> Logistic<T, Deterministic> {
  pub fn seeded(a: T, b: T, n: usize, x0: Option<T>, t: Option<T>, seed: u64) -> Self {
    Self {
      a,
      b,
      n,
      x0,
      t,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Logistic<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut logistic = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return logistic;
    }

    logistic[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return logistic;
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let sqrt_dt = dt.sqrt();
    let mut prev = logistic[0];
    let mut tail_view = logistic.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("Logistic output tail must be contiguous");
    let mut seed = self.seed;
    let normal = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &mut seed);
    normal.fill_slice_fast(tail);

    for z in tail.iter_mut() {
      let next = prev + prev * (T::one() - self.a * prev) * dt + self.b * prev * *z;
      *z = next;
      prev = next;
    }

    logistic
  }
}

py_process_1d!(PyLogistic, Logistic,
  sig: (a, b, n, x0=None, t=None, seed=None, dtype=None),
  params: (a: f64, b: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
