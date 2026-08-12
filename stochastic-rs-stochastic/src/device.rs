//! Compile-time sampling backends.
//!
//! A process is parameterised by a backend marker `B` ([`Cpu`] is the default);
//! the [`Backend`] trait monomorphises `sample` / `sample_par` to that backend
//! with **no runtime branch**. Switch backend with the turbofish
//! `process.on::<CudaNative>()` — the marker must be in scope, and the GPU
//! markers only exist when their feature is compiled, so selecting an
//! unavailable backend is a compile error rather than a runtime fallback.

use ndarray::Array1;
use ndarray::parallel::prelude::*;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::noise::fgn::Fgn;
use crate::traits::FloatExt;
#[cfg(feature = "accelerate")]
use crate::traits::process::chunk_count;
#[cfg(feature = "accelerate")]
use crate::traits::process::chunk_lens;

/// CPU backend — the default `B` for every process.
#[derive(Clone, Copy)]
pub struct Cpu;

/// cudarc + cuFFT + NVRTC Philox.
#[cfg(feature = "cuda-native")]
#[derive(Clone, Copy)]
pub struct CudaNative;

/// cubecl Rust kernels (CUDA or wgpu, per the compiled `cubecl-*` runtime).
#[cfg(feature = "gpu")]
#[derive(Clone, Copy)]
pub struct CubeCl;

/// Hand-written MSL via the `metal` crate. f32 only — Apple GPUs lack f64.
#[cfg(feature = "metal")]
#[derive(Clone, Copy)]
pub struct MetalNative;

/// Apple vDSP / AMX (FFI system framework, macOS).
#[cfg(feature = "accelerate")]
#[derive(Clone, Copy)]
pub struct Accelerate;

/// A compile-time fGN sampling backend. Implemented by the marker types in this
/// module; `Fgn<T, S, B>` dispatches to `B` with zero runtime branching.
///
/// The `Send + Sync` supertraits let a backend-parameterised process satisfy
/// the `ProcessExt: Send + Sync` bound and be shared across rayon worker
/// threads — every marker is a zero-sized unit struct, so this is free.
///
/// ## Reproducibility per backend
///
/// `Fgn`/`Fbm` are backend-generic, so a caller cannot tell from the type
/// alone what a given `B` guarantees under a pinned [`Deterministic`
/// ](stochastic_rs_core::simd_rng::Deterministic) seed. Spelled out:
///
/// | Backend | `sample`/`sample_par` reproducible? |
/// |---|---|
/// | [`Cpu`] | Yes — same seed + same `m` ⇒ bit-identical output on any machine, under any rayon thread-pool size. |
/// | `Accelerate` (`accelerate` feature) | **Not bit-identical — measured, not assumed.** Seed *consumption* (which derived basis feeds which path) is thread-count independent, via the identical mechanism `Cpu` uses. But `vDSP_fft_zip`'s own floating-point output is not bit-stable across otherwise-identical calls: measured on Apple Silicon (M4 Max), 400 repeated calls across varied `(n, m)` on an idle system showed zero divergence, but the same sweep with all cores saturated by unrelated work showed 21/400 configurations diverge (worst relative difference `2.08e-3`) — consistent with the heterogeneous P-core/E-core scheduler dispatching the FFT to different core types across calls. `Cpu`, under the identical induced load, stayed bit-exact throughout. Treat `Accelerate` as reproducible-effort-only, the same tier as the GPU backends below — see `tests/deterministic_parallelism_accelerate.rs`. |
/// | `CudaNative` / `CubeCl` / `MetalNative` (`cuda-native` / `gpu` / `metal` features) | **Not guaranteed.** Each batch call draws one `u32`/`u64` value from the fGN's seed (`self.seed.rng()`) and hands it to the on-device kernel's own Philox/PCG-style RNG — so output is a function of the pinned seed and *not* of host thread-pool size (there is no host-side rayon fan-out inside `generate_batch` for these backends), but cross-run bit-identity across GPU driver versions, vendors, or even repeated runs on the same device is untested and not promised. Treat these three as reproducible-effort-only. **Exception: [`Fbm`](crate::process::fbm::Fbm) on any of these three backends is not even a function of the pinned seed** — see [`Fbm::sample_par`](crate::process::fbm::Fbm::sample_par)'s own doc. |
///
/// `generate`/`generate_batch`/`generate_pair`'s `seed: &S2` parameter is the
/// mechanism `Cpu`/`Accelerate` use for their guarantees above (`Accelerate`'s
/// covers seed consumption only, not vDSP's own arithmetic); the GPU backends
/// ignore it (they seed from `fgn.seed` instead, once per batch call) exactly
/// as documented per-method below.
pub trait Backend: Sized + Send + Sync {
  /// One fGN increment vector. The host-side `seed` drives the CPU/Accelerate
  /// path only; GPU backends use the fGN's internal RNG (`fgn.seed`) and
  /// ignore this parameter — see the trait doc's reproducibility table.
  fn generate<T: FloatExt, S: SeedExt, S2: SeedExt>(fgn: &Fgn<T, S, Self>, seed: &S2) -> Array1<T>;

  /// `m` fGN paths in one batched call, one [`Array1`] per path. `seed`
  /// drives the CPU/Accelerate path (see each backend's own impl for the
  /// mechanism that makes it thread-count independent); GPU backends ignore
  /// it — see the trait doc's reproducibility table.
  fn generate_batch<T: FloatExt, S: SeedExt, S2: SeedExt>(
    fgn: &Fgn<T, S, Self>,
    m: usize,
    seed: &S2,
  ) -> Vec<Array1<T>>;

  /// Two independent fGN paths in one pass. Default: a batch of two; [`Cpu`]
  /// overrides with the real/imag parts of a single circulant FFT (one FFT,
  /// two independent fields — Dietrich & Newsam). `seed` drives the
  /// CPU/Accelerate path only, exactly as in [`generate_batch`](Self::generate_batch).
  fn generate_pair<T: FloatExt, S: SeedExt, S2: SeedExt>(
    fgn: &Fgn<T, S, Self>,
    seed: &S2,
  ) -> (Array1<T>, Array1<T>) {
    let mut paths = Self::generate_batch(fgn, 2, seed);
    let second = paths.pop().expect("generate_batch(2) yields two paths");
    let first = paths.pop().expect("generate_batch(2) yields two paths");
    (first, second)
  }
}

impl Backend for Cpu {
  fn generate<T: FloatExt, S: SeedExt, S2: SeedExt>(fgn: &Fgn<T, S, Self>, seed: &S2) -> Array1<T> {
    fgn.sample_cpu_impl(seed)
  }

  /// Derives one basis per **path** (not per `ProcessExt`-style chunk)
  /// sequentially on the calling thread via `seed.derive()`, before handing
  /// the `m` (basis, path-index) pairs to rayon — so which physical thread
  /// ends up computing path `i` no longer changes which basis path `i`
  /// consumes, fixing the thread-count dependence, while every path still
  /// gets its own independent rayon leaf task exactly as before this fix.
  /// Deliberately **not** `ProcessExt::chunk_count`-chunked: each path's
  /// own `ndrustfft::ndfft_inplace_par` call is itself a nested rayon
  /// `Zip::par_for_each` region, and measurement showed grouping several
  /// paths per outer task (reusing one `SimdNormal` sequentially across
  /// the group, mirroring `ProcessExt::chunked_samplers`) roughly doubled
  /// wall time at `m = 1000` — repeated nested-rayon entry from a single
  /// worker thread contends more than spreading the same nested calls
  /// across independent outer tasks. One basis per path costs one extra
  /// `SimdNormal` construction per path versus the (rejected) chunked
  /// design, which is negligible next to an FFT.
  fn generate_batch<T: FloatExt, S: SeedExt, S2: SeedExt>(
    fgn: &Fgn<T, S, Self>,
    m: usize,
    seed: &S2,
  ) -> Vec<Array1<T>> {
    (0..m)
      .map(|_| seed.derive())
      .collect::<Vec<_>>()
      .into_par_iter()
      .map(|path_seed| {
        let mut normal = SimdNormal::<T>::new(T::zero(), T::one(), &path_seed);
        array1_from_fill(fgn.out_len, |out| fgn.fill_cpu(&mut normal, out))
      })
      // `Vec::into_par_iter()` → `.map()` is an `IndexedParallelIterator`,
      // so `.collect()` restores index order regardless of completion
      // order — path `i` is always path `i`, independent of scheduling.
      .collect()
  }

  fn generate_pair<T: FloatExt, S: SeedExt, S2: SeedExt>(
    fgn: &Fgn<T, S, Self>,
    seed: &S2,
  ) -> (Array1<T>, Array1<T>) {
    fgn.sample_pair_cpu_impl(seed)
  }
}

/// Generates a [`Backend`] impl for a GPU marker whose `$sampler` returns an
/// `Array2<T>` of `m` paths. Single-path `generate` takes the first row; the
/// host-side seed is unused (GPU backends carry their own RNG, seeded from
/// `fgn.seed` once per call — see the trait doc's reproducibility table).
/// Each marker and its impl are gated on the backend's feature.
macro_rules! gpu_backend {
  ($feat:literal, $marker:ident => $sampler:ident) => {
    #[cfg(feature = $feat)]
    impl Backend for $marker {
      fn generate<T: FloatExt, S: SeedExt, S2: SeedExt>(
        fgn: &Fgn<T, S, Self>,
        _seed: &S2,
      ) -> Array1<T> {
        fgn.$sampler(1).unwrap().row(0).to_owned()
      }

      fn generate_batch<T: FloatExt, S: SeedExt, S2: SeedExt>(
        fgn: &Fgn<T, S, Self>,
        m: usize,
        _seed: &S2,
      ) -> Vec<Array1<T>> {
        fgn
          .$sampler(m)
          .unwrap()
          .outer_iter()
          .map(|row| row.to_owned())
          .collect()
      }
    }
  };
}

gpu_backend!("cuda-native", CudaNative => sample_cuda_native_impl);
gpu_backend!("gpu", CubeCl => sample_gpu_impl);
gpu_backend!("metal", MetalNative => sample_metal_impl);

/// Accelerate (vDSP) runs on the CPU, so it gets the same reproducibility
/// guarantee as [`Cpu`], reached via `ProcessExt::chunk_count`-style
/// chunking rather than [`Cpu`]'s own per-path derivation (see that impl's
/// doc for why the two diverge: `Cpu`'s FFT call nests a nested rayon
/// region and measurably regressed under chunking, `vDSP_fft_zip` does
/// not — it is a plain FFI call, so grouping several into one thread's
/// sequential work costs nothing extra). `generate_batch` splits `m` into
/// `chunk_count(m)` chunks (capped at `MAX_CHUNKS`), derives one basis per
/// chunk sequentially on the calling thread, then hands each chunk to rayon
/// as a single `sample_accelerate_impl(len, ..)` vDSP batch call — the
/// per-thread FFT setup and scratch are still cached and reused exactly as
/// before; only the granularity at which rayon schedules work changed (one
/// task per chunk instead of one task per path), which does not change
/// wall-clock throughput once `chunk_count(m)` meets or exceeds the core
/// count (see `MAX_CHUNKS`'s doc).
#[cfg(feature = "accelerate")]
impl Backend for Accelerate {
  fn generate<T: FloatExt, S: SeedExt, S2: SeedExt>(fgn: &Fgn<T, S, Self>, seed: &S2) -> Array1<T> {
    fgn
      .sample_accelerate_impl(1, seed)
      .unwrap()
      .row(0)
      .to_owned()
  }

  fn generate_batch<T: FloatExt, S: SeedExt, S2: SeedExt>(
    fgn: &Fgn<T, S, Self>,
    m: usize,
    seed: &S2,
  ) -> Vec<Array1<T>> {
    if m == 0 {
      return Vec::new();
    }
    let chunks = chunk_count(m);
    let chunk_seeds = (0..chunks).map(|_| seed.derive()).collect::<Vec<_>>();
    chunk_lens(m, chunks)
      .zip(chunk_seeds)
      .collect::<Vec<_>>()
      .into_par_iter()
      .map(|(len, chunk_seed)| {
        fgn
          .sample_accelerate_impl(len, &chunk_seed)
          .unwrap()
          .outer_iter()
          .map(|row| row.to_owned())
          .collect::<Vec<_>>()
      })
      .collect::<Vec<_>>()
      .into_iter()
      .flatten()
      .collect()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn cpu_marker_is_a_backend() {
    fn assert_backend<B: Backend>() {}
    assert_backend::<Cpu>();
  }
}
