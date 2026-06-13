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

pub use core::Fgn;

use ndarray::Array1;
#[cfg(feature = "python")]
pub use python::PyFgn;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Backend;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
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
  type Sampler<'s>
    = FgnSampler<'s, T, S, B>
  where
    Self: 's;

  /// A CPU sampler reusing the process's `Arc`-shared FFT plan and
  /// eigenvalues plus an owned Gaussian source. Note: even for GPU backends
  /// this samples on the CPU — GPU users should batch through
  /// [`sample_par`](Self::sample_par), which dispatches to the backend.
  fn sampler(&self) -> FgnSampler<'_, T, S, B> {
    FgnSampler {
      fgn: self,
      normal: SimdNormal::<T>::new(T::zero(), T::one(), &self.seed),
    }
  }

  fn sample(&self) -> Self::Output {
    B::generate(self, &self.seed)
  }

  /// The `m` paths are generated in **one batched backend call** (a single FFT
  /// plan over the whole batch).
  fn sample_par(&self, m: usize) -> Vec<Self::Output> {
    B::generate_batch(self, m)
  }
}

/// Reusable CPU fGn sampler: borrows the process (for its FFT plan and
/// eigenvalues) and owns the Gaussian source so a Monte-Carlo loop pays the
/// `SimdNormal` setup once.
#[doc(hidden)]
pub struct FgnSampler<'a, T: FloatExt, S: SeedExt, B> {
  fgn: &'a Fgn<T, S, B>,
  normal: SimdNormal<T>,
}

impl<T: FloatExt, S: SeedExt, B: Backend> PathSampler<T> for FgnSampler<'_, T, S, B> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Fgn output must be contiguous");
    self.fgn.fill_cpu(&mut self.normal, slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let out_len = self.fgn.out_len;
    array1_from_fill(out_len, |out| self.fgn.fill_cpu(&mut self.normal, out))
  }
}
