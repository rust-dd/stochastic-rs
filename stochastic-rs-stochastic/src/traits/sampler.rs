//! `PathSampler` — reusable per-thread sampling state.

use stochastic_rs_distributions::traits::FloatExt;

/// Mutable sampling state detached from the immutable process definition.
///
/// Implementation detail behind [`ProcessExt`](super::ProcessExt)'s
/// `sample` / `sample_par` / `sample_map`; not part of the public surface.
/// Owns what `sample()` used to rebuild per call: distribution state
/// (`SimdNormal`, Poisson drivers, …), precomputed scales, FFT scratch, and
/// sub-samplers for wrapper processes. `sample_into` overwrites a
/// caller-owned buffer with no allocation and no RNG setup — the regime where
/// short-path Monte Carlo measured 1.8–3× over per-path `sample()` calls.
///
/// Constructing a sampler derives its RNG exactly like a single `sample()`
/// call used to, so with [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic)
/// seeding the first `sample_into` reproduces the legacy stream bit-for-bit;
/// subsequent calls continue the stream, yielding independent paths.
#[doc(hidden)]
pub trait PathSampler<T: FloatExt>: Send {
  type Output: Send;

  /// Overwrites `out` with a fresh realisation.
  fn sample_into(&mut self, out: &mut Self::Output);

  /// One-shot sample: allocates the output and fills it.
  fn sample(&mut self) -> Self::Output;
}
