//! # fGN
//!
//! $$
//! \operatorname{Cov}(\Delta B_i^H,\Delta B_j^H)=\tfrac12\left(|k+1|^{2H}-2|k|^{2H}+|k-1|^{2H}\right),\ k=i-j
//! $$
//!
mod core;
#[cfg(feature = "cuda-native")]
mod cuda_native;
#[cfg(feature = "gpu")]
mod gpu;
#[cfg(feature = "python")]
mod python;

pub use core::FGN;

#[cfg(any(feature = "gpu", feature = "cuda-native"))]
use anyhow::Result;
#[cfg(any(feature = "gpu", feature = "cuda-native"))]
use either::Either;
use ndarray::Array1;
#[cfg(any(feature = "gpu", feature = "cuda-native"))]
use ndarray::Array2;
#[cfg(feature = "python")]
pub use python::PyFGN;

use crate::simd_rng::SeedExt;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

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
}
