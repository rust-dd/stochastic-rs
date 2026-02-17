mod core;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "python")]
mod python;

pub use core::FGN;

#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use either::Either;
use ndarray::Array1;
#[cfg(feature = "cuda")]
use ndarray::Array2;
#[cfg(feature = "python")]
pub use python::PyFGN;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

impl<T: FloatExt> ProcessExt<T> for FGN<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    self.sample_cpu()
  }

  #[cfg(feature = "cuda")]
  fn sample_cuda(&self, m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    self.sample_cuda_impl(m)
  }
}
