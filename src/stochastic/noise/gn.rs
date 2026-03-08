//! # GN
//!
//! $$
//! \Delta W_i\sim\mathcal N(0,\Delta t)
//! $$
//!
use ndarray::Array1;

use crate::distributions::normal::SimdNormal;
use crate::simd_rng::Deterministic;
use crate::simd_rng::Seed;
use crate::simd_rng::Unseeded;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

#[derive(Copy, Clone)]
pub struct Gn<T: FloatExt, S: Seed = Unseeded> {
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt> Gn<T> {
  pub fn new(n: usize, t: Option<T>) -> Self {
    Gn {
      n,
      t,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> Gn<T, Deterministic> {
  pub fn seeded(n: usize, t: Option<T>, seed: u64) -> Self {
    Gn {
      n,
      t,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: Seed> ProcessExt<T> for Gn<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut out = Array1::<T>::zeros(self.n);
    let out_slice = out.as_slice_mut().expect("Gn output must be contiguous");
    self.fill_slice(out_slice);
    out
  }
}

impl<T: FloatExt, S: Seed> Gn<T, S> {
  pub fn fill_slice(&self, out: &mut [T]) {
    let len = self.n.min(out.len());
    if len == 0 {
      return;
    }
    let std_dev = self.dt().sqrt();
    let mut seed = self.seed;
    let normal = SimdNormal::<T>::from_seed_source(T::zero(), std_dev, &mut seed);
    normal.fill_slice_fast(&mut out[..len]);
  }

  pub fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n)
  }
}

py_process_1d!(PyGn, Gn,
  sig: (n, t=None, dtype=None),
  params: (n: usize, t: Option<f64>)
);
