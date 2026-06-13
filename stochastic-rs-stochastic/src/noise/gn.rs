//! # GN
//!
//! $$
//! \Delta W_i\sim\mathcal N(0,\Delta t)
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

#[derive(Copy, Clone)]
pub struct Gn<T: FloatExt, S: SeedExt = Unseeded> {
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Gn<T, S> {
  pub fn new(n: usize, t: Option<T>, seed: S) -> Self {
    Gn { n, t, seed }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Gn<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = GnSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> GnSampler<T> {
    GnSampler {
      n: self.n,
      normal: SimdNormal::<T>::new(T::zero(), self.dt().sqrt(), &self.seed),
    }
  }
}

/// Reusable [`Gn`] sampling state: the owned Gaussian source. Each path is `n`
/// i.i.d. `N(0, dt)` increments (no leading zero, unlike [`Bm`]).
#[doc(hidden)]
pub struct GnSampler<T: FloatExt> {
  n: usize,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> GnSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    let len = self.n.min(out.len());
    if len == 0 {
      return;
    }
    self.normal.fill_slice_fast(&mut out[..len]);
  }
}

impl<T: FloatExt> PathSampler<T> for GnSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Gn output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

impl<T: FloatExt, S: SeedExt> Gn<T, S> {
  pub fn fill_slice(&self, out: &mut [T]) {
    let len = self.n.min(out.len());
    if len == 0 {
      return;
    }
    let std_dev = self.dt().sqrt();
    let normal = SimdNormal::<T>::new(T::zero(), std_dev, &self.seed);
    normal.fill_slice_fast(&mut out[..len]);
  }

  pub fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n)
  }
}

py_process_1d!(PyGn, Gn,
  sig: (n, t=None, seed=None, dtype=None),
  params: (n: usize, t: Option<f64>)
);
