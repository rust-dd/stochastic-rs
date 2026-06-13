//! # WN
//!
//! $$
//! \xi_i\stackrel{iid}{\sim}\mathcal N(0,1)
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
pub struct Wn<T: FloatExt, S: SeedExt = Unseeded> {
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Target mean level for generated noise samples.
  pub mean: Option<T>,
  /// Standard deviation of generated noise samples.
  pub std_dev: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Wn<T, S> {
  pub fn new(n: usize, mean: Option<T>, std_dev: Option<T>, seed: S) -> Self {
    Wn {
      n,
      mean,
      std_dev,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Wn<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = WnSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> WnSampler<T> {
    let mean = self.mean.unwrap_or(T::zero());
    let std_dev = self.std_dev.unwrap_or(T::one());
    WnSampler {
      n: self.n,
      normal: SimdNormal::<T>::new(mean, std_dev, &self.seed),
    }
  }
}

/// Reusable [`Wn`] sampling state: the owned Gaussian source. Each path is `n`
/// i.i.d. `N(mean, std_dev^2)` draws.
#[doc(hidden)]
pub struct WnSampler<T: FloatExt> {
  n: usize,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> WnSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    let len = self.n.min(out.len());
    if len == 0 {
      return;
    }
    self.normal.fill_slice_fast(&mut out[..len]);
  }
}

impl<T: FloatExt> PathSampler<T> for WnSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Wn output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyWn, Wn,
  sig: (n, mean=None, std_dev=None, seed=None, dtype=None),
  params: (n: usize, mean: Option<f64>, std_dev: Option<f64>)
);
