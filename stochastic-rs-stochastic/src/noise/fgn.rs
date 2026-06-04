//! # fGN
//!
//! $$
//! \operatorname{Cov}(\Delta B_i^H,\Delta B_j^H)=\tfrac12\left(|k+1|^{2H}-2|k|^{2H}+|k-1|^{2H}\right),\ k=i-j
//! $$
//!
#[cfg(feature = "accelerate")]
mod accelerate;
mod core;
#[cfg(feature = "cuda-native")]
mod cuda_native;
#[cfg(feature = "cuda-oxide-experimental")]
mod cuda_oxide;
#[cfg(feature = "gpu")]
mod gpu;
#[cfg(feature = "metal")]
mod metal;
#[cfg(feature = "python")]
mod python;

pub use core::Fgn;

use ndarray::Array1;
use ndarray::parallel::prelude::*;
#[cfg(feature = "python")]
pub use python::PyFgn;
use stochastic_rs_core::simd_rng::SeedExt;

use crate::device::Backend;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

impl<T: FloatExt, S: SeedExt, B> Fgn<T, S, B> {
  /// Sample two independent fGn paths in a single FFT / RNG pass (CPU).
  ///
  /// Both paths have the same covariance structure as `sample` and are
  /// statistically independent — Dietrich & Newsam (1997) and Kroese & Botev
  /// (2013 §2.2) identify the real and imaginary parts of the circulant-
  /// embedding FFT output as two independent realisations of the target field.
  /// For Monte-Carlo inner loops this ~halves wall time per path.
  pub fn sample_pair(&self) -> (Array1<T>, Array1<T>) {
    self.sample_pair_cpu()
  }

  /// As [`sample_pair`](Self::sample_pair) but with an explicit seed.
  pub fn sample_pair_with_seed(&self, seed: u64) -> (Array1<T>, Array1<T>) {
    self.sample_pair_cpu_with_seed(seed)
  }
}

impl<T: FloatExt, S: SeedExt, B: Backend> ProcessExt<T> for Fgn<T, S, B> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    B::generate(self, &self.seed)
  }

  /// The `m` paths are generated in **one batched backend call** (a single FFT
  /// plan over the whole batch), then copied out across all cores.
  fn sample_par(&self, m: usize) -> Vec<Self::Output> {
    B::generate_batch(self, m)
      .outer_iter()
      .into_par_iter()
      .map(|row| row.to_owned())
      .collect()
  }
}
