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

/// Circulant-embedding fGN from the caller's `seed`, not `fgn.seed`; bit-identical on [`Cpu`] only.
/// See <https://stochastic.rust-dd.com/docs/concepts/backends#reproducibility-per-backend>.
pub trait FgnBackend<T: FloatExt>: Backend {
  /// One fGN increment vector, or why the device could not produce it.
  fn try_generate<S: SeedExt, S2: SeedExt>(
    &self,
    fgn: &Fgn<T, S, Self>,
    seed: &S2,
  ) -> Result<Array1<T>, DeviceError>;

  /// `m` fGN paths in one batched call. The CPU devices derive one seed per path or chunk from
  /// `seed`; a GPU draws one launch seed from it per call, so a chunked batch equals one launch.
  fn try_generate_batch<S: SeedExt, S2: SeedExt>(
    &self,
    fgn: &Fgn<T, S, Self>,
    m: usize,
    seed: &S2,
  ) -> Result<Vec<Array1<T>>, DeviceError>;

  /// `f` over `m` fGN paths. The default owns each row first; a device overrides it to hand `f`
  /// the rows of its `m × n` block, saving a copy per path. The CPU devices' paths are owned.
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

/// Bit-identical for one seed and one `m`, on any machine and under any rayon thread-pool size.
impl<T: FloatExt> FgnBackend<T> for Cpu {
  fn try_generate<S: SeedExt, S2: SeedExt>(
    &self,
    fgn: &Fgn<T, S, Self>,
    seed: &S2,
  ) -> Result<Array1<T>, DeviceError> {
    Ok(fgn.sample_cpu_impl(seed))
  }

  /// One basis per path, derived before rayon, so no thread count changes a path's basis. Not
  /// chunked: the FFT nests a rayon region, and chunking doubled the wall time at `m = 1000`.
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
      // An indexed parallel iterator collects in index order whatever the completion order.
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

/// An [`FgnBackend`] impl for a GPU marker behind a bus, gated on its cfg: the map form walks the
/// rows of the batch once it is in host memory, which saves the default's copy per row.
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
          // The rows are read where the device's block lies; owning them first costs a copy each.
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

// Each device implements the capability for the scalars its kernels compute in, so `Fgn<f64>` on
// `Metal` is a compile error, not an `f32` computation behind an `f64` type.
gpu_backend!(feature = "cuda", Cuda => sample_cuda_impl, f32, f64);

/// Owned rows copy from shared Metal storage; map callbacks borrow it. Best effort, as on `Cuda`:
/// one seed gives one batch, but identical bits across drivers, devices or runs are not promised.
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

/// Seed consumption ignores the thread count (one basis per `chunk_count(m)` chunk), but vDSP is
/// not bit-stable: under full load 21 of 400 `(n, m)` sweeps diverged, worst `2.08e-3` (M4 Max).
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

  /// vDSP fills a chunk as one block, so the rows are read where they lie, not owned first.
  /// Chunks, per-chunk seeds and order are the batch form's: the same paths from the same streams.
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
