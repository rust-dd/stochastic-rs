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
#[cfg(feature = "gpu")]
mod gpu;
#[cfg(feature = "metal")]
mod metal;
#[cfg(feature = "python")]
mod python;

pub use core::FGN;

#[cfg(any(
  feature = "gpu",
  feature = "cuda-native",
  feature = "accelerate",
  feature = "metal"
))]
use anyhow::Result;
#[cfg(any(
  feature = "gpu",
  feature = "cuda-native",
  feature = "accelerate",
  feature = "metal"
))]
use either::Either;
use ndarray::Array1;
#[cfg(any(
  feature = "gpu",
  feature = "cuda-native",
  feature = "accelerate",
  feature = "metal"
))]
use ndarray::Array2;
#[cfg(feature = "python")]
pub use python::PyFGN;

use stochastic_rs_core::simd_rng::SeedExt;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

impl<T: FloatExt, S: SeedExt> FGN<T, S> {
  /// Sample two independent fGn paths in a single FFT / RNG pass.
  ///
  /// Both paths have the same covariance structure as [`sample`](Self::sample)
  /// and are statistically independent — Dietrich & Newsam (1997) and
  /// Kroese & Botev (2013 §2.2) identify the real and imaginary parts of
  /// the circulant-embedding FFT output as two independent realisations of
  /// the target Gaussian field.
  ///
  /// For Monte-Carlo inner loops this ~halves wall time per path vs.
  /// two back-to-back [`sample`](Self::sample) calls (one FFT / RNG pass
  /// reused for both outputs).
  pub fn sample_pair(&self) -> (Array1<T>, Array1<T>) {
    self.sample_pair_cpu()
  }

  /// As [`sample_pair`](Self::sample_pair) but with an explicit seed.
  pub fn sample_pair_with_seed(&self, seed: u64) -> (Array1<T>, Array1<T>) {
    self.sample_pair_cpu_with_seed(seed)
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for FGN<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    self.sample_cpu()
  }

  #[cfg(feature = "gpu")]
  fn sample_gpu(&self, m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    self.sample_gpu_impl(m)
  }

  #[cfg(feature = "cuda-native")]
  fn sample_cuda_native(&self, m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    self.sample_cuda_native_impl(m)
  }

  #[cfg(feature = "accelerate")]
  fn sample_accelerate(&self, m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    self.sample_accelerate_impl(m)
  }

  #[cfg(feature = "metal")]
  fn sample_metal(&self, m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    self.sample_metal_impl(m)
  }
}
