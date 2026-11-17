use ndarray::Array2;
use ndarray::parallel::prelude::*;
use stochastic_rs_core::simd_rng::SeedExt;

use super::Backend;
use super::error::DeviceError;
#[cfg(accelerate_backend)]
use super::handles::Accelerate;
use super::handles::Cpu;
#[cfg(feature = "cuda")]
use super::handles::Cuda;
#[cfg(metal_backend)]
use super::handles::Metal;
use crate::sheet::fbs::Fbs;
use crate::traits::FloatExt;

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
/// Reproducibility per backend is as [`FgnBackend`](super::FgnBackend) states it: a GPU draws
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
#[cfg(accelerate_backend)]
host_sheet_backend!(Accelerate);

/// Generates a [`SheetBackend`] impl for a GPU marker whose `$sampler` returns
/// the batch's sheets. Each marker and its impl are gated on the backend's
/// cfg, and each marker implements the capability for the scalars its
/// kernels compute in.
macro_rules! gpu_sheet_backend {
  ($cfg:meta, $marker:ident => $sampler:ident, $($scalar:ty),+) => {
    $(
      #[cfg($cfg)]
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

gpu_sheet_backend!(feature = "cuda", Cuda => sample_cuda_sheets, f32, f64);
gpu_sheet_backend!(metal_backend, Metal => sample_metal_sheets, f32);
