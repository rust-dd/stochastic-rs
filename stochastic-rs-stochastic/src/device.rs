//! Compile-time sampling backends.
//!
//! A process is parameterised by a backend marker `B` ([`Cpu`] is the default);
//! the [`Backend`] trait monomorphises `sample` / `sample_par` to that backend
//! with **no runtime branch**. Switch backend by handing `.on` a handle,
//! `process.on::<Cuda>()` — the marker must be in scope, and the GPU
//! markers only exist when their feature is compiled, so selecting an
//! unavailable backend is a compile error rather than a runtime fallback.
//!
//! The capability traits ([`FgnBackend`], [`crate::euler::EulerBackend`]) take
//! the scalar as a type parameter, and a device implements them only for the
//! precision its kernels compute in: `Cuda` for `f32` and `f64`,
//! `Metal` for `f32` alone. `Fgn<f64>` on `Metal`
//! does not compile; nothing is computed in `f32` behind an `f64` type.

use std::fmt;

use ndarray::Array1;
use ndarray::Array2;
use ndarray::parallel::prelude::*;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::noise::fgn::Fgn;
use crate::sheet::fbs::Fbs;
use crate::traits::FloatExt;
#[cfg(feature = "accelerate")]
use crate::traits::process::chunk_count;
#[cfg(feature = "accelerate")]
use crate::traits::process::chunk_lens;

/// CPU backend — the default `B` for every process.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cpu;

/// cudarc + cuFFT + NVRTC Philox.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cuda {
  /// Which device to open: the CUDA device ordinal.
  pub ordinal: usize,
  /// Bytes of path data one launch may hold; a larger batch runs as chunks
  /// whose union is bit-identical to one launch.
  pub batch_budget: usize,
}

#[cfg(feature = "cuda")]
impl Default for Cuda {
  /// Ordinal from `STOCHASTIC_RS_DEVICE` (else `0`), budget from
  /// `STOCHASTIC_RS_DEVICE_BATCH_BYTES` (else [`DEFAULT_BATCH_BUDGET_BYTES`]).
  fn default() -> Self {
    Self {
      ordinal: env_ordinal(),
      batch_budget: env_budget(),
    }
  }
}

#[cfg(feature = "cuda")]
impl Cuda {
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
pub struct Metal {
  /// Which device to open: the index into `Device::all()`, `0` being the system default.
  pub ordinal: usize,
  /// Bytes of path data one launch may hold; a larger batch runs as chunks
  /// whose union is bit-identical to one launch.
  pub batch_budget: usize,
}

#[cfg(feature = "metal")]
impl Default for Metal {
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
impl Metal {
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
  /// The device ran out of memory. Separate from [`Launch`](Self::Launch)
  /// because it is the one device failure a caller can do something about:
  /// the batch loops below halve their chunk and try again, and a chunk of
  /// this engine is bit-identical however the batch is cut, so the retry
  /// returns the same numbers rather than an approximation of them.
  OutOfMemory(String),
}

impl fmt::Display for DeviceError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      DeviceError::Unavailable(msg) => write!(f, "device unavailable: {msg}"),
      DeviceError::Compile(msg) => write!(f, "kernel compilation failed: {msg}"),
      DeviceError::Launch(msg) => write!(f, "device operation failed: {msg}"),
      DeviceError::OutOfMemory(msg) => write!(f, "device out of memory: {msg}"),
    }
  }
}

impl DeviceError {
  /// Whether the device failed for want of memory, which a smaller chunk may
  /// survive. Every other failure is the same however the batch is cut.
  pub fn is_out_of_memory(&self) -> bool {
    matches!(self, DeviceError::OutOfMemory(_))
  }
}

impl std::error::Error for DeviceError {}

/// What [`Backend::probe`] found behind a marker.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceInfo {
  /// The marker's name, e.g. `"Cuda"`.
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

// Read by the `Default` impls of the device handles, `cuda` and `metal`.
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
/// `STOCHASTIC_RS_DEVICE` parsed as an ordinal; anything unparsable is `0`.
pub(crate) fn device_from_env(value: Option<&str>) -> usize {
  value.and_then(|s| s.trim().parse().ok()).unwrap_or(0)
}

// Read by the `Default` impls of the device handles, `cuda` and `metal`.
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
/// The ordinal a device handle starts with: `STOCHASTIC_RS_DEVICE`, else `0`.
pub(crate) fn env_ordinal() -> usize {
  device_from_env(std::env::var("STOCHASTIC_RS_DEVICE").ok().as_deref())
}

/// Default cap on the path data one device launch materialises: 1 GiB.
pub const DEFAULT_BATCH_BUDGET_BYTES: usize = 1 << 30;

#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
/// `STOCHASTIC_RS_DEVICE_BATCH_BYTES` parsed; anything that is not a positive
/// number is the default.
pub(crate) fn budget_from_env(value: Option<&str>) -> usize {
  value
    .and_then(|s| s.trim().parse::<usize>().ok())
    .filter(|b| *b > 0)
    .unwrap_or(DEFAULT_BATCH_BUDGET_BYTES)
}

#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
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

/// Runs `chunk(first, len)` over `m` rows, halving the chunk each time the
/// device answers that it is out of memory, down to a single row.
///
/// The budget a handle carries is a guess — a fixed default, or a number the
/// caller chose — and a guess that is too large is otherwise a hard failure
/// on a device that would have served a smaller launch. Halving is safe here
/// and nowhere else in the crate: an engine chunk is bit-identical however
/// the batch is cut, so the retry produces the same paths.
pub(crate) fn over_chunks(
  m: usize,
  rows: usize,
  mut chunk: impl FnMut(usize, usize) -> Result<(), DeviceError>,
) -> Result<(), DeviceError> {
  let mut rows = rows.max(1);
  let mut first = 0;
  while first < m {
    let len = rows.min(m - first);
    match chunk(first, len) {
      Ok(()) => first += len,
      Err(e) if e.is_out_of_memory() && len > 1 => rows = len / 2,
      Err(e) => return Err(e),
    }
  }
  Ok(())
}

/// How many per-size device states (FFT plans, buffers) a back-end keeps.
/// Only the native CUDA and Metal fGN samplers cache per-size state, so a
/// build without them has no caller.
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
pub(crate) const CACHE_SLOTS: usize = 4;

/// The cached state matching `matches`, moved to the most-recent slot, or a
/// freshly built one after evicting the least-recent when the cache is full.
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
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

/// The panic a plain `sample*` call raises when its device fails.
pub(crate) fn device_panic<T>(e: DeviceError) -> T {
  panic!("{e}; probe the device handle with `Backend::probe(&device)` before sampling on it")
}

/// A device marker.
///
/// [`probe`](Self::probe) is the one run-time question a marker answers:
/// whether the device behind it can be used right now, and what it is. The
/// sampling itself stays a compile-time choice (`.on::<B>()`).
pub trait Backend: Copy + Send + Sync {
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
#[cfg(feature = "cuda")]
impl Backend for Cuda {
  fn probe(&self) -> Result<DeviceInfo, DeviceError> {
    crate::euler::cuda::probe(self.ordinal)
  }
}
#[cfg(feature = "metal")]
impl Backend for Metal {
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
/// | `Cuda` / `Metal` (`cuda` / `metal` features) | **Not guaranteed.** Each batch call draws one `u32`/`u64` value from the `seed: &S2` the caller passed — the process's own seed, so two `Deterministic` processes built from the same seed value produce the same device paths, and consecutive calls on one process advance the stream into independent paths, exactly as on the host — and hands it to the on-device kernel's own Philox/PCG-style RNG, with a per-chunk offset so a chunked batch equals one launch. Output is therefore a function of the pinned seed and *not* of host thread-pool size (no host-side rayon fan-out inside `generate_batch` for these backends), but cross-run bit-identity across GPU driver versions, vendors, or even repeated runs on the same device is untested and not promised. Treat these two as reproducible-effort-only. |
///
/// `generate`/`generate_batch`/`generate_pair`'s `seed: &S2` parameter is the
/// mechanism behind every row above: `Cpu`/`Accelerate` derive one basis per
/// path or chunk from it (`Accelerate`'s guarantee covers seed consumption
/// only, not vDSP's own arithmetic), and the GPU backends draw one launch
/// seed from it per batch. It is the caller's seed, not `fgn.seed`, which is
/// what makes a wrapper whose embedded `fgn` is [`Unseeded`
/// ](stochastic_rs_core::simd_rng::Unseeded) — [`Fbm`](crate::process::fbm::Fbm)
/// and the `Fou` family — reproducible on a device too.
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

  /// `f` over `m` fGN paths, mapped where the batch lies.
  ///
  /// A device produces the batch as one `m × n` block and the default here
  /// splits it into `m` owned arrays before the caller sees a single value —
  /// at ten thousand paths over four thousand points that is ten thousand
  /// allocations and a hundred and sixty megabytes copied, on whatever host
  /// the card is attached to, which is several times the transfer that
  /// preceded it. A device overrides this to hand `f` the rows themselves.
  ///
  /// The CPU devices have nothing to save: their paths are owned arrays from
  /// the start, so the default is what they should do.
  fn try_generate_map<S: SeedExt, S2: SeedExt, R: Send>(
    &self,
    fgn: &Fgn<T, S, Self>,
    m: usize,
    seed: &S2,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Result<Vec<R>, DeviceError> {
    Ok(
      self
        .try_generate_batch(fgn, m, seed)?
        .iter()
        .map(|path| f(path.view()))
        .collect(),
    )
  }

  /// [`try_generate_map`](Self::try_generate_map), panicking with the
  /// device's error.
  fn generate_map<S: SeedExt, S2: SeedExt, R: Send>(
    &self,
    fgn: &Fgn<T, S, Self>,
    m: usize,
    seed: &S2,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    self
      .try_generate_map(fgn, m, seed, f)
      .unwrap_or_else(device_panic)
  }

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

  /// Two paths from one batched call, the device's error instead of its
  /// panic.
  fn try_generate_pair<S: SeedExt, S2: SeedExt>(
    &self,
    fgn: &Fgn<T, S, Self>,
    seed: &S2,
  ) -> Result<(Array1<T>, Array1<T>), DeviceError> {
    let mut paths = self.try_generate_batch(fgn, 2, seed)?;
    let second = paths.pop().expect("generate_batch(2) yields two paths");
    let first = paths.pop().expect("generate_batch(2) yields two paths");
    Ok((first, second))
  }

  /// Two paths from one batched call.
  fn generate_pair<S: SeedExt, S2: SeedExt>(
    &self,
    fgn: &Fgn<T, S, Self>,
    seed: &S2,
  ) -> (Array1<T>, Array1<T>) {
    self
      .try_generate_pair(fgn, seed)
      .unwrap_or_else(device_panic)
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

  fn try_generate_pair<S: SeedExt, S2: SeedExt>(
    &self,
    fgn: &Fgn<T, S, Self>,
    seed: &S2,
  ) -> Result<(Array1<T>, Array1<T>), DeviceError> {
    Ok(fgn.sample_pair_cpu_impl(seed))
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

        fn try_generate_map<S: SeedExt, S2: SeedExt, R: Send>(
          &self,
          fgn: &Fgn<$scalar, S, Self>,
          m: usize,
          seed: &S2,
          f: impl Fn(ndarray::ArrayView1<$scalar>) -> R + Sync,
        ) -> Result<Vec<R>, DeviceError> {
          use rayon::prelude::*;
          let batch = fgn.$sampler(m, seed, self)?;
          // The rows of the block the device filled, read where they are.
          // Owning them first is what the batch form does, and on a slow
          // host that copy costs several times the transfer.
          Ok(match batch.as_slice() {
            Some(flat) => flat
              .par_chunks(fgn.out_len.max(1))
              .map(|row| f(ndarray::ArrayView1::from(row)))
              .collect(),
            None => batch
              .outer_iter()
              .collect::<Vec<_>>()
              .into_par_iter()
              .map(&f)
              .collect(),
          })
        }
      }
    )+
  };
}

// Each device implements the capability for the scalars its kernels compute
// in: the native CUDA kernels are templated on float and double, the Metal
// FFT pipeline is single precision. `Fgn<f64>` on `Metal` is therefore a
// compile error, not an `f32` computation behind an `f64` type.
gpu_backend!("cuda", Cuda => sample_cuda_impl, f32, f64);
gpu_backend!("metal", Metal => sample_metal_impl, f32);

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

  /// vDSP fills a chunk as one block, the same as a GPU does, so the rows are
  /// read where they lie rather than owned first — the one place an
  /// Accelerate batch was paying a copy a path.
  ///
  /// The chunking, the per-chunk seed derivation and the order they are
  /// flattened back in are the batch form's, unchanged: this maps the same
  /// paths, in the same order, from the same streams.
  fn try_generate_map<S: SeedExt, S2: SeedExt, R: Send>(
    &self,
    fgn: &Fgn<T, S, Self>,
    m: usize,
    seed: &S2,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Result<Vec<R>, DeviceError> {
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
            .map(&f)
            .collect::<Vec<_>>(),
        )
      })
      .collect::<Result<Vec<_>, DeviceError>>()
      .map(|chunks| chunks.into_iter().flatten().collect())
  }
}

/// The sheet-sampling capability of a [`Backend`]: the two-dimensional
/// circulant embedding behind [`Fbs`]. The host devices run the process's own
/// sampler; a GPU runs the embedding's noise transform as a pipeline of its
/// own — complex Gaussian noise scaled by the eigenvalues' square roots, a
/// row transform, a transpose, a column transform, and the read-out of the
/// leading block less its corner plus the low-rank correction — in `f32`
/// (Metal) or `f32` and `f64` (CUDA). The GPU pipelines take a grid
/// whose embedding sides `2(m − 1)` and `2(n − 1)` are powers of two; the
/// process keeps any other grid on the host itself.
///
/// Reproducibility per backend is as [`FgnBackend`] states it: a GPU draws
/// one launch seed per batch from the process's own seed and hashes every
/// normal from it, so a `Deterministic` sheet reproduces its device sheets
/// and a batch produced in chunks equals one launch sheet for sheet.
pub trait SheetBackend<T: FloatExt>: Backend {
  /// One sheet, or why the device could not produce it.
  fn try_sheet<S: SeedExt>(&self, fbs: &Fbs<T, S, Self>) -> Result<Array2<T>, DeviceError>;

  /// `m` sheets in one batched call, or why the device could not produce
  /// them.
  fn try_sheets<S: SeedExt>(
    &self,
    fbs: &Fbs<T, S, Self>,
    m: usize,
  ) -> Result<Vec<Array2<T>>, DeviceError>;

  /// `f` over `m` sheets, keeping the results rather than the sheets.
  fn try_sheets_map<S: SeedExt, R: Send>(
    &self,
    fbs: &Fbs<T, S, Self>,
    m: usize,
    f: impl Fn(&Array2<T>) -> R + Sync,
  ) -> Result<Vec<R>, DeviceError> {
    let sheets = self.try_sheets(fbs, m)?;
    let f = &f;
    Ok(sheets.par_iter().map(f).collect())
  }
}

/// The host devices sample a sheet through the process's own sampler,
/// chunked exactly as [`crate::traits::ProcessExt`] chunks, so a `Cpu` build
/// and a device build that falls back to the host agree to the bit.
macro_rules! host_sheet_backend {
  ($marker:ty) => {
    impl<T: FloatExt> SheetBackend<T> for $marker {
      fn try_sheet<S: SeedExt>(&self, fbs: &Fbs<T, S, Self>) -> Result<Array2<T>, DeviceError> {
        Ok(fbs.host_sheet())
      }

      fn try_sheets<S: SeedExt>(
        &self,
        fbs: &Fbs<T, S, Self>,
        m: usize,
      ) -> Result<Vec<Array2<T>>, DeviceError> {
        Ok(crate::traits::process::sample_par_chunked(fbs, m))
      }

      fn try_sheets_map<S: SeedExt, R: Send>(
        &self,
        fbs: &Fbs<T, S, Self>,
        m: usize,
        f: impl Fn(&Array2<T>) -> R + Sync,
      ) -> Result<Vec<R>, DeviceError> {
        Ok(crate::traits::process::sample_map_chunked(fbs, m, f))
      }
    }
  };
}

host_sheet_backend!(Cpu);
#[cfg(feature = "accelerate")]
host_sheet_backend!(Accelerate);

/// Generates a [`SheetBackend`] impl for a GPU marker whose `$sampler` returns
/// the batch's sheets. Each marker and its impl are gated on the backend's
/// feature, and each marker implements the capability for the scalars its
/// kernels compute in.
macro_rules! gpu_sheet_backend {
  ($feat:literal, $marker:ident => $sampler:ident, $($scalar:ty),+) => {
    $(
      #[cfg(feature = $feat)]
      impl SheetBackend<$scalar> for $marker {
        fn try_sheet<S: SeedExt>(
          &self,
          fbs: &Fbs<$scalar, S, Self>,
        ) -> Result<Array2<$scalar>, DeviceError> {
          Ok(
            fbs
              .$sampler(1, self)?
              .pop()
              .expect("one sheet was asked for"),
          )
        }

        fn try_sheets<S: SeedExt>(
          &self,
          fbs: &Fbs<$scalar, S, Self>,
          m: usize,
        ) -> Result<Vec<Array2<$scalar>>, DeviceError> {
          fbs.$sampler(m, self)
        }
      }
    )+
  };
}

gpu_sheet_backend!("cuda", Cuda => sample_cuda_sheets, f32, f64);
gpu_sheet_backend!("metal", Metal => sample_metal_sheets, f32);

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

  #[test]
  fn cpu_marker_has_the_sheet_capability() {
    fn assert_sheet<B: SheetBackend<f64>>() {}
    assert_sheet::<Cpu>();
  }
}
