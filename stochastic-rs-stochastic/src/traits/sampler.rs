//! `PathSampler` — reusable per-thread sampling state.

use stochastic_rs_distributions::traits::FloatExt;

use crate::device::DeviceError;

/// Mutable sampling state behind `ProcessExt::sample*`: the streams, scales and scratch one path
/// draws from; `sample_into` refills a caller buffer. Sealed and `#[doc(hidden)]`: an implementation detail.
#[doc(hidden)]
pub trait PathSampler<T: FloatExt>: Send + crate::traits::Sealed {
  type Output: Send;

  /// Overwrites `out` with a fresh realisation.
  fn sample_into(&mut self, out: &mut Self::Output);

  /// One-shot sample: allocates the output and fills it.
  fn sample(&mut self) -> Self::Output;

  /// [`sample`](Self::sample), reporting a device failure as a
  /// [`DeviceError`] instead of panicking. A sampler that never reaches a
  /// device keeps this default, which cannot fail.
  fn try_sample(&mut self) -> Result<Self::Output, DeviceError> {
    Ok(self.sample())
  }
}
