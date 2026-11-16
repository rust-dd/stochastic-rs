use ndarray::Array1;
use ndarray::parallel::prelude::*;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use super::Backend;
use super::chunks::device_panic;
use super::error::DeviceError;
#[cfg(accelerate_backend)]
use super::handles::Accelerate;
use super::handles::Cpu;
#[cfg(feature = "cuda")]
use super::handles::Cuda;
#[cfg(metal_backend)]
use super::handles::Metal;
use crate::buffer::array1_from_fill;
use crate::noise::fgn::Fgn;
use crate::traits::FloatExt;
#[cfg(accelerate_backend)]
use crate::traits::process::chunk_count;
#[cfg(accelerate_backend)]
use crate::traits::process::chunk_lens;

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
        let mut normal = SimdNormal::<T>::new(T::zero(), T::one()).seeded(&path_seed);
        array1_from_fill(fgn.n(), |out| fgn.fill_cpu(&mut normal, out))
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

/// Owns rows returned by the CUDA sampler, using the caller's seed source.
#[cfg(feature = "cuda")]
macro_rules! gpu_backend_owning {
  ($sampler:ident, $scalar:ty) => {
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
  };
}

/// Generates an [`FgnBackend`] impl for a GPU marker behind a bus, whose
/// batch has to cross into host memory before anything can read it: the map
/// form therefore takes that batch and walks its rows, which still saves the
/// per-row copy the trait's default would add. Each marker and its impl are
/// gated on the backend's cfg.
macro_rules! gpu_backend {
  ($cfg:meta, $marker:ident => $sampler:ident, $($scalar:ty),+) => {
    $(
      #[cfg($cfg)]
      impl FgnBackend<$scalar> for $marker {
        gpu_backend_owning!($sampler, $scalar);

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
              .par_chunks(fgn.n().max(1))
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
gpu_backend!(feature = "cuda", Cuda => sample_cuda_impl, f32, f64);

/// Owned rows copy directly from shared Metal storage; map callbacks borrow it.
#[cfg(metal_backend)]
impl FgnBackend<f32> for Metal {
  fn try_generate<S: SeedExt, S2: SeedExt>(
    &self,
    fgn: &Fgn<f32, S, Self>,
    seed: &S2,
  ) -> Result<Array1<f32>, DeviceError> {
    Ok(
      fgn
        .sample_metal_impl(1, seed, self)?
        .pop()
        .expect("one requested row"),
    )
  }

  fn try_generate_batch<S: SeedExt, S2: SeedExt>(
    &self,
    fgn: &Fgn<f32, S, Self>,
    m: usize,
    seed: &S2,
  ) -> Result<Vec<Array1<f32>>, DeviceError> {
    fgn.sample_metal_impl(m, seed, self)
  }

  fn try_generate_map<S: SeedExt, S2: SeedExt, R: Send>(
    &self,
    fgn: &Fgn<f32, S, Self>,
    m: usize,
    seed: &S2,
    f: impl Fn(ndarray::ArrayView1<f32>) -> R + Sync,
  ) -> Result<Vec<R>, DeviceError> {
    fgn.map_metal_impl(m, seed, self, f)
  }
}

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
#[cfg(accelerate_backend)]
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
