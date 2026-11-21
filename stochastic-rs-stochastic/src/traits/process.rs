//! `ProcessExt` and dimensional output markers.

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::parallel::prelude::*;
use stochastic_rs_distributions::traits::FloatExt;

use super::sampler::PathSampler;
use crate::device::DeviceError;

mod chunks;
mod markers;

pub(crate) use chunks::chunk_count;
pub(crate) use chunks::chunk_lens;
pub(crate) use chunks::sample_map_chunked;
pub(crate) use chunks::sample_par_chunked;
pub(crate) use chunks::try_sample_par_chunked;
pub use markers::ComplexPathOutput;
pub use markers::CurveOutput;
pub use markers::MultiDimensional;
pub use markers::OneDimensional;
pub use markers::TwoDimensional;
pub use markers::VariableDimensional;

/// Samples a process. On `Cpu` one seed and one `m` give bit-identical output on any thread pool;
/// `Clone` snapshots the seed. See <https://stochastic.rust-dd.com/docs/concepts/process-ext>.
///
/// ```
/// use stochastic_rs_core::simd_rng::Deterministic;
/// use stochastic_rs_stochastic::diffusion::ou::Ou;
/// use stochastic_rs_stochastic::traits::ProcessExt;
///
/// let a = Ou::<f64, _>::new(0.5, 0.02, 0.1, 32, Some(0.03), Some(1.0), Deterministic::new(42));
/// let b = a.clone();
/// // `b` snapshots `a`'s seed exactly as it stood at `.clone()`, so sampled
/// // immediately afterward — before either side draws anything else — the
/// // two agree bit-for-bit.
/// assert_eq!(a.sample(), b.sample());
/// ```
pub trait ProcessExt<T: FloatExt>: Send + Sync + crate::traits::Sealed {
  type Output: Send;

  /// Reusable sampling state. Implementation detail of the `sample*` methods,
  /// not part of the public surface.
  #[doc(hidden)]
  type Sampler<'a>: PathSampler<T, Output = Self::Output>
  where
    Self: 'a;

  /// Builds the per-call state. It must take its basis with `self.seed.derive()`: a `clone()` gives
  /// chunks overlapping streams, and reading `&self.seed` per path races in the parallel region.
  #[doc(hidden)]
  fn sampler(&self) -> Self::Sampler<'_>;

  /// Called once per chunk before `sampler()` and once after `sample`; a no-op by default. Override
  /// it to tick the seed where `sampler()` clones it into an engine a chunk reuses (`CirPlusPlus`).
  #[doc(hidden)]
  fn advance_chunk_seed(&self) {}

  /// Why this configuration cannot run on a device kernel (a grid past a per-path array, a closure
  /// coefficient, …), or `None`. A fallback is an order of magnitude slower and otherwise silent.
  ///
  /// ```ignore
  /// if let Some(why) = process.device_fallback() {
  ///   eprintln!("sampling on the host: {why}");
  /// }
  /// ```
  fn device_fallback(&self) -> Option<&'static str> {
    None
  }

  /// Whether this configuration runs on a device backend, i.e. `device_fallback()` is `None`.
  /// A `false` never fails: on any backend the process samples on the host, bit-identical to `Cpu`.
  fn device_ready(&self) -> bool {
    self.device_fallback().is_none()
  }

  /// One sampler per chunk with its path count, built sequentially on the calling thread before
  /// rayon runs, so the order in which the seed is consumed does not depend on the scheduler.
  #[doc(hidden)]
  fn chunked_samplers(&self, m: usize) -> Vec<(Self::Sampler<'_>, usize)> {
    let chunks = chunk_count(m);
    chunk_lens(m, chunks)
      .map(|len| {
        self.advance_chunk_seed();
        (self.sampler(), len)
      })
      .collect()
  }

  /// A single sampled path. `advance_chunk_seed` ticks after sampling, not before: a clone-based
  /// `sampler()` reads the seed's current state, so a tick first would skip a state.
  fn sample(&self) -> Self::Output {
    let out = self.sampler().sample();
    self.advance_chunk_seed();
    out
  }

  /// Maps `f` over `m` independent paths, reusing one sampler and one output buffer per chunk.
  /// One seed and one `m` give bit-identical output under any rayon thread-pool size.
  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Self::Output) -> R + Sync) -> Vec<R> {
    sample_map_chunked(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<Self::Output> {
    sample_par_chunked(self, m)
  }

  /// [`sample_map`](Self::sample_map) with a borrowed row, for single-row processes: same values,
  /// no copy out of the device's `m × n` batch, which is worth 1.4-2.1x on Metal.
  fn sample_map_view<R: Send>(&self, m: usize, f: impl Fn(ArrayView1<T>) -> R + Sync) -> Vec<R>
  where
    Self: ProcessExt<T, Output = Array1<T>> + Sized,
  {
    self.sample_map(m, |path| f(path.view()))
  }

  /// [`sample`](Self::sample) that returns a device failure as a [`DeviceError`] instead of
  /// panicking; on the host devices it cannot fail and the path is `sample`'s.
  fn try_sample(&self) -> Result<Self::Output, DeviceError> {
    let out = self.sampler().try_sample();
    self.advance_chunk_seed();
    out
  }

  /// [`sample_par`](Self::sample_par) that returns a device failure as a [`DeviceError`] instead
  /// of panicking; on the host devices it cannot fail and the paths are `sample_par`'s bit for bit.
  fn try_sample_par(&self, m: usize) -> Result<Vec<Self::Output>, DeviceError> {
    try_sample_par_chunked(self, m)
  }
}
