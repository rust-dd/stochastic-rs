//! # WN
//!
//! $$
//! \xi_i\stackrel{iid}{\sim}\mathcal N(0,1)
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
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

impl<T: FloatExt> Wn<T> {
  pub fn new(n: usize, mean: Option<T>, std_dev: Option<T>) -> Self {
    Wn {
      n,
      mean,
      std_dev,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> Wn<T, Deterministic> {
  pub fn seeded(n: usize, mean: Option<T>, std_dev: Option<T>, seed: u64) -> Self {
    Wn {
      n,
      mean,
      std_dev,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Wn<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mean = self.mean.unwrap_or(T::zero());
    let std_dev = self.std_dev.unwrap_or(T::one());
    let mut out = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return out;
    }
    let out_slice = out.as_slice_mut().expect("Wn output must be contiguous");
    let mut seed = self.seed;
    let normal = SimdNormal::<T>::from_seed_source(mean, std_dev, &mut seed);
    normal.fill_slice_fast(out_slice);
    out
  }
}

py_process_1d!(PyWn, Wn,
  sig: (n, mean=None, std_dev=None, seed=None, dtype=None),
  params: (n: usize, mean: Option<f64>, std_dev: Option<f64>)
);
