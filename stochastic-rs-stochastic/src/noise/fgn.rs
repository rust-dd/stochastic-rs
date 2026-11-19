//! Fractional Gaussian noise by Davies–Harte circulant embedding, on the host or a device.
//!
//! $$
//! \operatorname{Cov}(\Delta B_i^H,\Delta B_j^H)
//! =\tfrac12\left(|k+1|^{2H}-2|k|^{2H}+|k-1|^{2H}\right),\ k=i-j
//! $$
//!
#[cfg(accelerate_backend)]
mod accelerate;
#[cfg(accelerate_backend)]
pub(crate) use accelerate::release_accelerate_scratch;
mod core;
#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(metal_backend)]
pub(crate) mod metal;
#[cfg(feature = "python")]
mod python;

pub use core::Fgn;

use ndarray::Array1;
#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyFgn;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::device::DeviceError;
use crate::device::FgnBackend;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

impl<T: FloatExt, S: SeedExt, B> Fgn<T, S, B> {
  /// Two independent fGn paths from one FFT and RNG pass (CPU), at about half the cost per path:
  /// the output's real and imaginary parts, Dietrich & Newsam (1997), Kroese & Botev (2013 §2.2).
  pub fn sample_pair(&self) -> (Array1<T>, Array1<T>) {
    self.sample_pair_cpu()
  }
}

impl<T: FloatExt> Fgn<T, Unseeded, Cpu> {
  /// The fGN of a grid of `n` points: its `n − 1` increments.
  pub(crate) fn for_grid(hurst: T, n: usize, t: Option<T>) -> Self {
    Self::new(hurst, n - 1, t, Unseeded)
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> crate::traits::Sealed for Fgn<T, S, B> {}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> ProcessExt<T> for Fgn<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = FgnSampler<'s, T, S, B>
  where
    Self: 's;

  /// A CPU sampler on the process's shared FFT plan and eigenvalues, with its own Gaussian source.
  /// It samples on the CPU for a GPU backend too; batch through [`sample_par`](Self::sample_par).
  fn sampler(&self) -> FgnSampler<'_, T, S, B> {
    FgnSampler {
      fgn: self,
      normal: SimdNormal::<T>::new(T::zero(), T::one()).seeded(self.seed()),
    }
  }

  fn sample(&self) -> Self::Output {
    self.backend.generate(self, self.seed())
  }

  /// `m` paths in one batched backend call. On [`Cpu`](crate::device::Cpu) the same seed and `m`
  /// give identical bits at any thread count; [`FgnBackend`](crate::device::FgnBackend) has more.
  fn sample_par(&self, m: usize) -> Vec<Self::Output> {
    self.backend.generate_batch(self, m, self.seed())
  }

  /// Through the batched call [`sample_par`](Self::sample_par) takes; the trait's defaults would
  /// sample path by path on the host, a different stream from the device's pipeline.
  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    self
      .backend
      .generate_map(self, m, self.seed(), |row| f(&row.to_owned()))
  }

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    self.backend.generate_map(self, m, self.seed(), f)
  }

  fn try_sample(&self) -> Result<Self::Output, DeviceError> {
    self.backend.try_generate(self, self.seed())
  }

  /// [`sample_par`](Self::sample_par) with the device's error instead of its panic; on the CPU
  /// devices always `Ok` and bit-identical to `sample_par`.
  fn try_sample_par(&self, m: usize) -> Result<Vec<Self::Output>, DeviceError> {
    self.backend.try_generate_batch(self, m, self.seed())
  }
}

/// Reusable CPU fGn sampler: borrows the process's FFT plan and eigenvalues and owns the Gaussian
/// source.
#[doc(hidden)]
#[derive(Debug)]
pub struct FgnSampler<'a, T: FloatExt, S: SeedExt, B> {
  fgn: &'a Fgn<T, S, B>,
  normal: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> crate::traits::Sealed for FgnSampler<'_, T, S, B> {}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> PathSampler<T> for FgnSampler<'_, T, S, B> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Fgn output must be contiguous");
    self.fgn.fill_cpu(&mut self.normal, slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let out_len = self.fgn.n();
    array1_from_fill(out_len, |out| self.fgn.fill_cpu(&mut self.normal, out))
  }
}

#[cfg(test)]
mod tests {
  use super::Fgn;

  #[test]
  fn a_grid_of_n_points_has_n_minus_one_increments() {
    assert_eq!(Fgn::<f64>::for_grid(0.7, 65, None).n(), 64);
  }
}
