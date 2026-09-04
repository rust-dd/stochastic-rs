//! Compile-time sampling backends.
//!
//! A process is parameterised by a backend marker `B` ([`Cpu`] is the default);
//! the [`Backend`] trait monomorphises `sample` / `sample_par` to that backend
//! with **no runtime branch**. Switch backend with the turbofish
//! `process.on::<CudaNative>()` — the marker must be in scope, and the GPU
//! markers only exist when their feature is compiled, so selecting an
//! unavailable backend is a compile error rather than a runtime fallback.
//!
//! The capability traits ([`FgnBackend`], [`crate::euler::EulerBackend`]) take
//! the scalar as a type parameter, and a device implements them only for the
//! precision its kernels compute in: `CudaNative` for `f32` and `f64`,
//! `MetalNative` and `CubeCl` for `f32` alone. `Fgn<f64>` on `MetalNative`
//! does not compile; nothing is computed in `f32` behind an `f64` type.

use std::fmt;

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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cpu;

/// cudarc + cuFFT + NVRTC Philox.
#[cfg(feature = "cuda-native")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CudaNative {
  /// Which device to open: the CUDA device ordinal.
  pub ordinal: usize,
  /// Bytes of path data one launch may hold; a larger batch runs as chunks
  /// whose union is bit-identical to one launch.
  pub batch_budget: usize,
}

#[cfg(feature = "cuda-native")]
impl Default for CudaNative {
  /// Ordinal from `STOCHASTIC_RS_DEVICE` (else `0`), budget from
  /// `STOCHASTIC_RS_DEVICE_BATCH_BYTES` (else [`DEFAULT_BATCH_BUDGET_BYTES`]).
  fn default() -> Self {
    Self {
      ordinal: env_ordinal(),
      batch_budget: env_budget(),
    }
  }
}

#[cfg(feature = "cuda-native")]
impl CudaNative {
  /// The device at `ordinal` with the default batch budget.
  pub fn new(ordinal: usize) -> Self {
    Self {
      ordinal,
      ..Self::default()
    }
  }

  /// The same device with `bytes` of path data per launch.
  pub fn with_batch_budget(self, bytes: usize) -> Self {
    Self {
      batch_budget: bytes.max(1),
      ..self
    }
  }
}

/// cubecl Rust kernels (CUDA or wgpu, per the compiled `cubecl-*` runtime).
#[cfg(feature = "cubecl")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CubeCl {
  /// Which device to open: the runtime's device index (with `cubecl-wgpu`, `0` is the default adapter and `n > 0` the n-th discrete GPU).
  pub ordinal: usize,
  /// Bytes of path data one launch may hold; a larger batch runs as chunks
  /// whose union is bit-identical to one launch.
  pub batch_budget: usize,
}

#[cfg(feature = "cubecl")]
impl Default for CubeCl {
  /// Ordinal from `STOCHASTIC_RS_DEVICE` (else `0`), budget from
  /// `STOCHASTIC_RS_DEVICE_BATCH_BYTES` (else [`DEFAULT_BATCH_BUDGET_BYTES`]).
  fn default() -> Self {
    Self {
      ordinal: env_ordinal(),
      batch_budget: env_budget(),
    }
  }
}

#[cfg(feature = "cubecl")]
impl CubeCl {
  /// The device at `ordinal` with the default batch budget.
  pub fn new(ordinal: usize) -> Self {
    Self {
      ordinal,
      ..Self::default()
    }
  }

  /// The same device with `bytes` of path data per launch.
  pub fn with_batch_budget(self, bytes: usize) -> Self {
    Self {
      batch_budget: bytes.max(1),
      ..self
    }
  }
}

/// Hand-written MSL via the `metal` crate. f32 only — Apple GPUs lack f64.
#[cfg(feature = "metal")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalNative {
  /// Which device to open: the index into `Device::all()`, `0` being the system default.
  pub ordinal: usize,
  /// Bytes of path data one launch may hold; a larger batch runs as chunks
  /// whose union is bit-identical to one launch.
  pub batch_budget: usize,
}

#[cfg(feature = "metal")]
impl Default for MetalNative {
  /// Ordinal from `STOCHASTIC_RS_DEVICE` (else `0`), budget from
  /// `STOCHASTIC_RS_DEVICE_BATCH_BYTES` (else [`DEFAULT_BATCH_BUDGET_BYTES`]).
  fn default() -> Self {
    Self {
      ordinal: env_ordinal(),
      batch_budget: env_budget(),
    }
  }
}

#[cfg(feature = "metal")]
impl MetalNative {
  /// The device at `ordinal` with the default batch budget.
  pub fn new(ordinal: usize) -> Self {
    Self {
      ordinal,
      ..Self::default()
    }
  }

  /// The same device with `bytes` of path data per launch.
  pub fn with_batch_budget(self, bytes: usize) -> Self {
    Self {
      batch_budget: bytes.max(1),
      ..self
    }
  }
}

/// Apple vDSP / AMX (FFI system framework, macOS).
#[cfg(feature = "accelerate")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Accelerate;

/// A compile-time device marker. Implemented by every marker type in this
/// module; a process parameterised by `B: Backend` monomorphises to that
/// device with zero runtime branching.
///
/// The trait itself carries **no algorithm** — what a device can actually
/// compute is expressed by capability subtraits ([`FgnBackend`] is the first;
/// a future accelerated path engine adds its own without touching this
/// trait or any implementor). Bounding on `Backend` says "this type is
/// device-parameterised"; bounding on a capability says what the device
/// must know how to do.
///
/// The `Send + Sync` supertraits let a backend-parameterised process satisfy
/// the `ProcessExt: Send + Sync` bound and be shared across rayon worker
/// threads — every marker is a zero-sized unit struct, so this is free.
/// Why a device could not serve a request.
///
/// Returned by [`Backend::probe`] and the `try_*` device calls
/// ([`FgnBackend::try_generate_batch`],
/// [`crate::euler::EulerBackend::try_euler_paths`], `try_sample_par` on the
/// device-capable processes). The plain `sample*` calls panic with the same
/// message when the device fails, so probing first turns an environmental
/// failure into a `Result` instead of a panic.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeviceError {
  /// No usable device behind the marker, or its runtime failed to initialise.
  Unavailable(String),
  /// The kernel source did not compile for this device.
  Compile(String),
  /// A kernel launch, allocation or copy failed at run time.
  Launch(String),
}

impl fmt::Display for DeviceError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      DeviceError::Unavailable(msg) => write!(f, "device unavailable: {msg}"),
      DeviceError::Compile(msg) => write!(f, "kernel compilation failed: {msg}"),
      DeviceError::Launch(msg) => write!(f, "device operation failed: {msg}"),
    }
  }
}

impl std::error::Error for DeviceError {}

/// What [`Backend::probe`] found behind a marker.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceInfo {
  /// The marker's name, e.g. `"CudaNative"`.
  pub backend: &'static str,
  /// The device's own name, e.g. `"NVIDIA A100-SXM4-40GB"` or `"Apple M2 Max"`.
  pub name: String,
  /// Scalars the device computes in, e.g. `["f32"]` for an Apple GPU.
  pub precisions: &'static [&'static str],
  /// Device ordinal for back-ends that enumerate devices, `None` otherwise.
  pub ordinal: Option<usize>,
}

impl DeviceInfo {
  pub(crate) fn new(
    backend: &'static str,
    name: String,
    precisions: &'static [&'static str],
    ordinal: Option<usize>,
  ) -> Self {
    Self {
      backend,
      name,
      precisions,
      ordinal,
    }
  }

  fn host(backend: &'static str, what: &str) -> Self {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    Self::new(
      backend,
      format!("{what}, {threads} threads"),
      &["f32", "f64"],
      None,
    )
  }
}

#[cfg_attr(
  not(any(feature = "cuda-native", feature = "metal", feature = "cubecl")),
  allow(dead_code)
)]
/// `STOCHASTIC_RS_DEVICE` parsed as an ordinal; anything unparsable is `0`.
pub(crate) fn device_from_env(value: Option<&str>) -> usize {
  value.and_then(|s| s.trim().parse().ok()).unwrap_or(0)
}

#[cfg_attr(
  not(any(feature = "cuda-native", feature = "metal", feature = "cubecl")),
  allow(dead_code)
)]
/// The ordinal a device handle starts with: `STOCHASTIC_RS_DEVICE`, else `0`.
pub(crate) fn env_ordinal() -> usize {
  device_from_env(std::env::var("STOCHASTIC_RS_DEVICE").ok().as_deref())
}

/// Default cap on the path data one device launch materialises: 1 GiB.
pub const DEFAULT_BATCH_BUDGET_BYTES: usize = 1 << 30;

#[cfg_attr(
  not(any(feature = "cuda-native", feature = "metal", feature = "cubecl")),
  allow(dead_code)
)]
/// `STOCHASTIC_RS_DEVICE_BATCH_BYTES` parsed; anything that is not a positive
/// number is the default.
pub(crate) fn budget_from_env(value: Option<&str>) -> usize {
  value
    .and_then(|s| s.trim().parse::<usize>().ok())
    .filter(|b| *b > 0)
    .unwrap_or(DEFAULT_BATCH_BUDGET_BYTES)
}

#[cfg_attr(
  not(any(feature = "cuda-native", feature = "metal", feature = "cubecl")),
  allow(dead_code)
)]
/// The batch budget a device handle starts with: `STOCHASTIC_RS_DEVICE_BATCH_BYTES`,
/// else [`DEFAULT_BATCH_BUDGET_BYTES`].
pub(crate) fn env_budget() -> usize {
  budget_from_env(
    std::env::var("STOCHASTIC_RS_DEVICE_BATCH_BYTES")
      .ok()
      .as_deref(),
  )
}

/// Paths of `n` `elem`-byte scalars that fit `budget`, at least one.
pub(crate) fn chunk_rows(budget: usize, n: usize, elem: usize) -> usize {
  (budget / (n.max(1) * elem.max(1))).max(1)
}

/// How many per-size device states (FFT plans, buffers) a back-end keeps.
/// Only the native CUDA and Metal fGN samplers cache per-size state, so a
/// build without them has no caller.
#[cfg_attr(not(any(feature = "cuda-native", feature = "metal")), allow(dead_code))]
pub(crate) const CACHE_SLOTS: usize = 4;

/// The cached state matching `matches`, moved to the most-recent slot, or a
/// freshly built one after evicting the least-recent when the cache is full.
#[cfg_attr(not(any(feature = "cuda-native", feature = "metal")), allow(dead_code))]
pub(crate) fn lru_slot<C, E>(
  cache: &mut Vec<C>,
  matches: impl Fn(&C) -> bool,
  build: impl FnOnce() -> Result<C, E>,
) -> Result<&mut C, E> {
  if let Some(i) = cache.iter().position(&matches) {
    let hit = cache.remove(i);
    cache.push(hit);
  } else {
    // Build first: a failed build must not evict anything.
    let built = build()?;
    if cache.len() >= CACHE_SLOTS {
      cache.remove(0);
    }
    cache.push(built);
  }
  Ok(cache.last_mut().expect("the slot was just pushed"))
}

/// The text of a caught panic payload, for runtimes that panic instead of
/// returning an error when no device is present.
#[cfg(any(feature = "cubecl-cuda", feature = "cubecl-wgpu"))]
pub(crate) fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
  if let Some(s) = payload.downcast_ref::<&str>() {
    (*s).to_string()
  } else if let Some(s) = payload.downcast_ref::<String>() {
    s.clone()
  } else {
    "the runtime panicked while opening the device".to_string()
  }
}

/// The panic a plain `sample*` call raises when its device fails.
pub(crate) fn device_panic<T>(e: DeviceError) -> T {
  panic!("{e}; probe the device handle with `Backend::probe(&device)` before sampling on it")
}

/// A device marker.
///
/// [`probe`](Self::probe) is the one run-time question a marker answers:
/// whether the device behind it can be used right now, and what it is. The
/// sampling itself stays a compile-time choice (`.on::<B>()`).
pub trait Backend: Copy + Default + Send + Sync {
  /// Opens the device behind this marker and describes it, or says why it
  /// cannot be used (no device, runtime missing, kernels failing to
  /// compile). The CPU devices are always `Ok`. A `sample*` call on a device
  /// that fails this probe panics with the same error; the `try_*` calls
  /// return it.
  fn probe(&self) -> Result<DeviceInfo, DeviceError>;
}

impl Backend for Cpu {
  fn probe(&self) -> Result<DeviceInfo, DeviceError> {
    Ok(DeviceInfo::host("Cpu", "host CPU (SIMD)"))
  }
}
#[cfg(feature = "cuda-native")]
impl Backend for CudaNative {
  fn probe(&self) -> Result<DeviceInfo, DeviceError> {
    crate::euler::cuda_native::probe(self.ordinal)
  }
}
#[cfg(feature = "cubecl")]
impl Backend for CubeCl {
  fn probe(&self) -> Result<DeviceInfo, DeviceError> {
    #[cfg(any(feature = "cubecl-cuda", feature = "cubecl-wgpu"))]
    {
      crate::euler::gpu::probe(self.ordinal)
    }

    #[cfg(not(any(feature = "cubecl-cuda", feature = "cubecl-wgpu")))]
    {
      Err(DeviceError::Unavailable(
        "no CubeCL runtime compiled; enable the cubecl-cuda or cubecl-wgpu feature".to_string(),
      ))
    }
  }
}
#[cfg(feature = "metal")]
impl Backend for MetalNative {
  fn probe(&self) -> Result<DeviceInfo, DeviceError> {
    crate::euler::metal::probe(self.ordinal)
  }
}
#[cfg(feature = "accelerate")]
impl Backend for Accelerate {
  fn probe(&self) -> Result<DeviceInfo, DeviceError> {
    Ok(DeviceInfo::host("Accelerate", "host CPU (Apple vDSP)"))
  }
}

/// Host capability: the process samples on the CPU through its own
/// [`ProcessExt`](crate::traits::ProcessExt) sampler. Every process carries a
/// backend parameter `B` and accepts at least the host devices in `on::<B2>()`;
/// a process whose bound is this trait has no device kernel yet, and gaining
/// one later only widens the bound (to [`crate::euler::EulerBackend`] or
/// [`FgnBackend`]), which breaks no caller. [`Cpu`] and `Accelerate` (vDSP, a
/// CPU device) implement it.
pub trait HostBackend: Backend {}

impl HostBackend for Cpu {}

#[cfg(feature = "accelerate")]
impl HostBackend for Accelerate {}

/// The fGN sampling capability of a [`Backend`]: circulant-embedding
/// fractional Gaussian noise, the one algorithm every device implements
/// today. `Fgn<T, S, B>` dispatches to `B` through this trait.
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
/// | `CudaNative` / `CubeCl` / `MetalNative` (`cuda-native` / `gpu` / `metal` features) | **Not guaranteed.** Each batch call draws one `u32`/`u64` value from the fGN's seed (`self.seed.rng()`) and hands it to the on-device kernel's own Philox/PCG-style RNG — so output is a function of the pinned seed and *not* of host thread-pool size (there is no host-side rayon fan-out inside `generate_batch` for these backends), but cross-run bit-identity across GPU driver versions, vendors, or even repeated runs on the same device is untested and not promised. Treat these three as reproducible-effort-only. **Exception: [`Fbm`](crate::process::fbm::Fbm) on any of these three backends is not even a function of the pinned seed** — see [`Fbm::sample_par`](crate::process::fbm::Fbm)'s own doc. |
///
/// `generate`/`generate_batch`/`generate_pair`'s `seed: &S2` parameter is the
/// mechanism `Cpu`/`Accelerate` use for their guarantees above (`Accelerate`'s
/// covers seed consumption only, not vDSP's own arithmetic); the GPU backends
/// ignore it (they seed from `fgn.seed` instead, once per batch call) exactly
/// as documented per-method below.
pub trait FgnBackend<T: FloatExt>: Backend {
  /// One fGN increment vector, or why the device could not produce it.
  fn try_generate<S: SeedExt, S2: SeedExt>(
    &self,
    fgn: &Fgn<T, S, Self>,
    seed: &S2,
  ) -> Result<Array1<T>, DeviceError>;

  /// `m` fGN paths in one batched call, one [`Array1`] per path, or why the
  /// device could not produce them. The CPU devices derive one seed per path
  /// from `seed`; the GPU devices draw their launch seed from it once per call,
  /// so the caller's seed source (a wrapper's own `Deterministic`, say) is what
  /// reproduces device paths.
  fn try_generate_batch<S: SeedExt, S2: SeedExt>(
    &self,
    fgn: &Fgn<T, S, Self>,
    m: usize,
    seed: &S2,
  ) -> Result<Vec<Array1<T>>, DeviceError>;

  /// [`try_generate`](Self::try_generate), panicking with the device's error.
  fn generate<S: SeedExt, S2: SeedExt>(&self, fgn: &Fgn<T, S, Self>, seed: &S2) -> Array1<T> {
    self.try_generate(fgn, seed).unwrap_or_else(device_panic)
  }

  /// [`try_generate_batch`](Self::try_generate_batch), panicking with the
  /// device's error.
  fn generate_batch<S: SeedExt, S2: SeedExt>(
    &self,
    fgn: &Fgn<T, S, Self>,
    m: usize,
    seed: &S2,
  ) -> Vec<Array1<T>> {
    self
      .try_generate_batch(fgn, m, seed)
      .unwrap_or_else(device_panic)
  }

  /// Two paths from one batched call.
  fn generate_pair<S: SeedExt, S2: SeedExt>(
    &self,
    fgn: &Fgn<T, S, Self>,
    seed: &S2,
  ) -> (Array1<T>, Array1<T>) {
    let mut paths = self.generate_batch(fgn, 2, seed);
    let second = paths.pop().expect("generate_batch(2) yields two paths");
    let first = paths.pop().expect("generate_batch(2) yields two paths");
    (first, second)
  }
}

impl<T: FloatExt> FgnBackend<T> for Cpu {
  fn try_generate<S: SeedExt, S2: SeedExt>(
    &self,
    fgn: &Fgn<T, S, Self>,
    seed: &S2,
  ) -> Result<Array1<T>, DeviceError> {
    Ok(fgn.sample_cpu_impl(seed))
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
  fn try_generate_batch<S: SeedExt, S2: SeedExt>(
    &self,
    fgn: &Fgn<T, S, Self>,
    m: usize,
    seed: &S2,
  ) -> Result<Vec<Array1<T>>, DeviceError> {
    let paths = (0..m)
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
      .collect();
    Ok(paths)
  }

  fn generate_pair<S: SeedExt, S2: SeedExt>(
    &self,
    fgn: &Fgn<T, S, Self>,
    seed: &S2,
  ) -> (Array1<T>, Array1<T>) {
    fgn.sample_pair_cpu_impl(seed)
  }
}

/// Generates an [`FgnBackend`] impl for a GPU marker whose `$sampler` returns an
/// `Array2<T>` of `m` paths. Single-path `generate` takes the first row. The
/// caller's seed source drives the launch seed, exactly as on the CPU: a
/// wrapper such as `Fbm` keeps an `Unseeded` inner `Fgn` and hands over its
/// own seed, so a `Deterministic` wrapper reproduces its device paths too.
/// Each marker and its impl are gated on the backend's feature.
macro_rules! gpu_backend {
  ($feat:literal, $marker:ident => $sampler:ident, $($scalar:ty),+) => {
    $(
      #[cfg(feature = $feat)]
      impl FgnBackend<$scalar> for $marker {
        fn try_generate<S: SeedExt, S2: SeedExt>(
          &self,
          fgn: &Fgn<$scalar, S, Self>,
          seed: &S2,
        ) -> Result<Array1<$scalar>, DeviceError> {
          Ok(fgn.$sampler(1, seed, self)?.row(0).to_owned())
        }

        fn try_generate_batch<S: SeedExt, S2: SeedExt>(
          &self,
          fgn: &Fgn<$scalar, S, Self>,
          m: usize,
          seed: &S2,
        ) -> Result<Vec<Array1<$scalar>>, DeviceError> {
          Ok(
            fgn
              .$sampler(m, seed, self)?
              .outer_iter()
              .map(|row| row.to_owned())
              .collect(),
          )
        }
      }
    )+
  };
}

// Each device implements the capability for the scalars its kernels compute
// in: the native CUDA kernels are templated on float and double, the Metal
// and CubeCL FFT pipelines are single precision. `Fgn<f64>` on `MetalNative`
// is therefore a compile error, not an `f32` computation behind an `f64` type.
gpu_backend!("cuda-native", CudaNative => sample_cuda_native_impl, f32, f64);
gpu_backend!("cubecl", CubeCl => sample_gpu_impl, f32);
gpu_backend!("metal", MetalNative => sample_metal_impl, f32);

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
impl<T: FloatExt> FgnBackend<T> for Accelerate {
  fn try_generate<S: SeedExt, S2: SeedExt>(
    &self,
    fgn: &Fgn<T, S, Self>,
    seed: &S2,
  ) -> Result<Array1<T>, DeviceError> {
    Ok(fgn.sample_accelerate_impl(1, seed)?.row(0).to_owned())
  }

  fn try_generate_batch<S: SeedExt, S2: SeedExt>(
    &self,
    fgn: &Fgn<T, S, Self>,
    m: usize,
    seed: &S2,
  ) -> Result<Vec<Array1<T>>, DeviceError> {
    if m == 0 {
      return Ok(Vec::new());
    }
    let chunks = chunk_count(m);
    let chunk_seeds = (0..chunks).map(|_| seed.derive()).collect::<Vec<_>>();
    chunk_lens(m, chunks)
      .zip(chunk_seeds)
      .collect::<Vec<_>>()
      .into_par_iter()
      .map(|(len, chunk_seed)| {
        Ok(
          fgn
            .sample_accelerate_impl(len, &chunk_seed)?
            .outer_iter()
            .map(|row| row.to_owned())
            .collect::<Vec<_>>(),
        )
      })
      .collect::<Result<Vec<_>, DeviceError>>()
      .map(|chunks| chunks.into_iter().flatten().collect())
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

  /// The marker trait alone must stay algorithm-free: this compiles because
  /// `Cpu` has the fGN capability, and a future device that lacks it can
  /// still be a [`Backend`] for the capabilities it does have.
  #[test]
  fn lru_slot_keeps_four_and_promotes_hits() {
    let mut cache: Vec<usize> = Vec::new();
    for k in 0..4 {
      lru_slot::<usize, ()>(&mut cache, |c| *c == k, || Ok(k)).unwrap();
    }
    assert_eq!(cache, vec![0, 1, 2, 3]);
    lru_slot::<usize, ()>(&mut cache, |c| *c == 1, || Ok(1)).unwrap();
    assert_eq!(
      cache,
      vec![0, 2, 3, 1],
      "a hit moves to the most-recent slot"
    );
    lru_slot::<usize, ()>(&mut cache, |c| *c == 9, || Ok(9)).unwrap();
    assert_eq!(
      cache,
      vec![2, 3, 1, 9],
      "a miss evicts the least-recent slot"
    );
    assert!(lru_slot::<usize, ()>(&mut cache, |c| *c == 7, || Err(())).is_err());
    assert_eq!(
      cache,
      vec![2, 3, 1, 9],
      "a failed build leaves the cache alone"
    );
  }

  #[test]
  fn batch_budget_parses_the_environment_leniently() {
    assert_eq!(budget_from_env(None), DEFAULT_BATCH_BUDGET_BYTES);
    assert_eq!(budget_from_env(Some("0")), DEFAULT_BATCH_BUDGET_BYTES);
    assert_eq!(budget_from_env(Some("x")), DEFAULT_BATCH_BUDGET_BYTES);
    assert_eq!(budget_from_env(Some(" 4096 ")), 4096);
  }

  #[test]
  fn device_ordinal_parses_the_environment_leniently() {
    assert_eq!(device_from_env(None), 0);
    assert_eq!(device_from_env(Some(" 2 ")), 2);
    assert_eq!(device_from_env(Some("gpu1")), 0);
    assert_eq!(device_from_env(Some("")), 0);
  }

  #[test]
  fn cpu_probe_reports_both_precisions() {
    let info = Cpu.probe().expect("the host is always available");
    assert_eq!(info.backend, "Cpu");
    assert_eq!(info.precisions, &["f32", "f64"]);
    assert_eq!(info.ordinal, None);
  }

  #[test]
  fn device_error_names_its_kind() {
    assert_eq!(
      DeviceError::Unavailable("no Metal device".into()).to_string(),
      "device unavailable: no Metal device"
    );
    assert_eq!(
      DeviceError::Compile("NVRTC euler_paths_float: x".into()).to_string(),
      "kernel compilation failed: NVRTC euler_paths_float: x"
    );
    assert_eq!(
      DeviceError::Launch("alloc out: y".into()).to_string(),
      "device operation failed: alloc out: y"
    );
  }

  #[test]
  fn cpu_marker_has_the_fgn_capability() {
    fn assert_fgn<B: FgnBackend<f64>>() {}
    assert_fgn::<Cpu>();
  }
}
