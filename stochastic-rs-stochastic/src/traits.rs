//! # Stochastic process traits
//!
//! `ProcessExt` for sampling, `MalliavinExt` / `Malliavin2DExt` for finite-
//! difference Malliavin sensitivities.

use ndarray::Array1;
#[cfg(any(
  feature = "gpu",
  feature = "cuda-native",
  feature = "accelerate",
  feature = "metal"
))]
use ndarray::Array2;
use ndarray::parallel::prelude::*;

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

// Re-export upstream traits so `crate::traits::FloatExt` inside this sub-crate
// continues to resolve in moved code.
#[cfg(feature = "python")]
pub use stochastic_rs_distributions::traits::CallableDist;
pub use stochastic_rs_distributions::traits::DistributionExt;
pub use stochastic_rs_distributions::traits::DistributionSampler;
pub use stochastic_rs_distributions::traits::FloatExt;
pub use stochastic_rs_distributions::traits::Fn1D;
pub use stochastic_rs_distributions::traits::Fn2D;
pub use stochastic_rs_distributions::traits::SimdFloatExt;

use crate::noise::gn::Gn;

pub trait ProcessExt<T: FloatExt>: Send + Sync {
  type Output: Send;

  fn sample(&self) -> Self::Output;

  fn sample_par(&self, m: usize) -> Vec<Self::Output> {
    (0..m).into_par_iter().map(|_| self.sample()).collect()
  }

  #[cfg(feature = "gpu")]
  fn sample_gpu(&self, _m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    anyhow::bail!("CubeCL GPU sampling is not supported for this process")
  }

  #[cfg(feature = "cuda-native")]
  fn sample_cuda_native(&self, _m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    anyhow::bail!("cudarc native CUDA sampling is not supported for this process")
  }

  #[cfg(feature = "accelerate")]
  fn sample_accelerate(&self, _m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    anyhow::bail!("Accelerate/vDSP sampling is not supported for this process")
  }

  #[cfg(feature = "metal")]
  fn sample_metal(&self, _m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    anyhow::bail!("Metal GPU sampling is not supported for this process")
  }
}

pub trait MalliavinExt<T: FloatExt> {
  fn sample_with_noise(&self, noise: &Array1<T>) -> Array1<T>;

  fn n(&self) -> usize;

  fn t(&self) -> Option<T>;

  fn malliavin_derivative<F>(&self, f: F, epsilon: T) -> Array1<T>
  where
    F: Fn(&Array1<T>) -> T,
  {
    let gn = Gn::new(self.n() - 1, self.t());
    let mut noise = gn.sample();
    let path = self.sample_with_noise(&noise);
    let f_original = f(&path);
    let mut derivatives = Array1::zeros(noise.len());

    for i in 0..noise.len() {
      let original = noise[i];
      noise[i] += epsilon;
      let path_perturbed = self.sample_with_noise(&noise);
      derivatives[i] = (f(&path_perturbed) - f_original) / epsilon;
      noise[i] = original;
    }

    derivatives
  }

  fn malliavin_derivative_terminal(&self, epsilon: T) -> Array1<T> {
    self.malliavin_derivative(|path| *path.last().unwrap(), epsilon)
  }
}

pub trait Malliavin2DExt<T: FloatExt> {
  fn sample_with_noise(&self, noise: &[Array1<T>; 2]) -> [Array1<T>; 2];

  fn generate_noise(&self) -> [Array1<T>; 2];

  fn malliavin_derivative<F>(&self, f: F, epsilon: T, noise_component: usize) -> Array1<T>
  where
    F: Fn(&[Array1<T>; 2]) -> T,
  {
    let mut noise = self.generate_noise();
    let paths = self.sample_with_noise(&noise);
    let f_original = f(&paths);
    let n = noise[noise_component].len();
    let mut derivatives = Array1::zeros(n);

    for i in 0..n {
      let original = noise[noise_component][i];
      noise[noise_component][i] += epsilon;
      let paths_perturbed = self.sample_with_noise(&noise);
      derivatives[i] = (f(&paths_perturbed) - f_original) / epsilon;
      noise[noise_component][i] = original;
    }

    derivatives
  }

  fn malliavin_derivative_terminal(
    &self,
    epsilon: T,
    path_component: usize,
    noise_component: usize,
  ) -> Array1<T> {
    self.malliavin_derivative(
      |paths| *paths[path_component].last().unwrap(),
      epsilon,
      noise_component,
    )
  }
}
