//! Compile-time sampling backends.
//!
//! A process is parameterised by a backend marker `B` ([`Cpu`] is the default);
//! the [`Backend`] trait monomorphises `sample` / `sample_par` to that backend
//! with **no runtime branch**. Switch backend with the turbofish
//! `process.on::<CudaNative>()` — the marker must be in scope, and the GPU
//! markers only exist when their feature is compiled, so selecting an
//! unavailable backend is a compile error rather than a runtime fallback.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::parallel::prelude::*;
use stochastic_rs_core::simd_rng::SeedExt;

use crate::noise::fgn::Fgn;
use crate::traits::FloatExt;

/// CPU backend — the default `B` for every process.
pub struct Cpu;

/// cudarc + cuFFT + NVRTC Philox.
#[cfg(feature = "cuda-native")]
pub struct CudaNative;

/// cuda-oxide Rust → PTX.
#[cfg(feature = "cuda-oxide-experimental")]
pub struct CudaOxide;

/// cubecl Rust kernels (CUDA or wgpu, per the compiled `cubecl-*` runtime).
#[cfg(feature = "gpu")]
pub struct CubeCl;

/// Hand-written MSL via the `metal` crate. f32 only — Apple GPUs lack f64.
#[cfg(feature = "metal")]
pub struct MetalNative;

/// Apple vDSP / AMX (FFI system framework, macOS).
#[cfg(feature = "accelerate")]
pub struct Accelerate;

/// A compile-time fGN sampling backend. Implemented by the marker types in this
/// module; `Fgn<T, S, B>` dispatches to `B` with zero runtime branching.
///
/// The `Send + Sync` supertraits let a backend-parameterised process satisfy
/// the `ProcessExt: Send + Sync` bound and be shared across rayon worker
/// threads — every marker is a zero-sized unit struct, so this is free.
pub trait Backend: Sized + Send + Sync {
  /// One fGN increment vector. The host-side `seed` drives the CPU path only;
  /// GPU backends use the fGN's internal RNG.
  fn generate<T: FloatExt, S: SeedExt, S2: SeedExt>(fgn: &Fgn<T, S, Self>, seed: &S2) -> Array1<T>;

  /// `m` fGN paths in one batched call, returned `m × n` row-major.
  fn generate_batch<T: FloatExt, S: SeedExt>(fgn: &Fgn<T, S, Self>, m: usize) -> Array2<T>;
}

impl Backend for Cpu {
  fn generate<T: FloatExt, S: SeedExt, S2: SeedExt>(fgn: &Fgn<T, S, Self>, seed: &S2) -> Array1<T> {
    fgn.sample_cpu_impl(seed)
  }

  fn generate_batch<T: FloatExt, S: SeedExt>(fgn: &Fgn<T, S, Self>, m: usize) -> Array2<T> {
    let rows: Vec<Array1<T>> = (0..m).into_par_iter().map(|_| fgn.sample_cpu()).collect();
    stack_rows(rows)
  }
}

fn stack_rows<T: FloatExt>(rows: Vec<Array1<T>>) -> Array2<T> {
  let n = rows.first().map_or(0, Array1::len);
  let mut out = Array2::<T>::zeros((rows.len(), n));
  for (i, row) in rows.iter().enumerate() {
    out.row_mut(i).assign(row);
  }
  out
}

#[cfg(feature = "cuda-native")]
impl Backend for CudaNative {
  fn generate<T: FloatExt, S: SeedExt, S2: SeedExt>(fgn: &Fgn<T, S, Self>, _seed: &S2) -> Array1<T> {
    first_row(fgn.sample_cuda_native_impl(1), "CudaNative")
  }
  fn generate_batch<T: FloatExt, S: SeedExt>(fgn: &Fgn<T, S, Self>, m: usize) -> Array2<T> {
    to_batch(fgn.sample_cuda_native_impl(m), "CudaNative")
  }
}

#[cfg(feature = "cuda-oxide-experimental")]
impl Backend for CudaOxide {
  fn generate<T: FloatExt, S: SeedExt, S2: SeedExt>(fgn: &Fgn<T, S, Self>, _seed: &S2) -> Array1<T> {
    first_row(fgn.sample_cuda_oxide_impl(1), "CudaOxide")
  }
  fn generate_batch<T: FloatExt, S: SeedExt>(fgn: &Fgn<T, S, Self>, m: usize) -> Array2<T> {
    to_batch(fgn.sample_cuda_oxide_impl(m), "CudaOxide")
  }
}

#[cfg(feature = "gpu")]
impl Backend for CubeCl {
  fn generate<T: FloatExt, S: SeedExt, S2: SeedExt>(fgn: &Fgn<T, S, Self>, _seed: &S2) -> Array1<T> {
    first_row(fgn.sample_gpu_impl(1), "CubeCl")
  }
  fn generate_batch<T: FloatExt, S: SeedExt>(fgn: &Fgn<T, S, Self>, m: usize) -> Array2<T> {
    to_batch(fgn.sample_gpu_impl(m), "CubeCl")
  }
}

#[cfg(feature = "metal")]
impl Backend for MetalNative {
  fn generate<T: FloatExt, S: SeedExt, S2: SeedExt>(fgn: &Fgn<T, S, Self>, _seed: &S2) -> Array1<T> {
    first_row(fgn.sample_metal_impl(1), "MetalNative")
  }
  fn generate_batch<T: FloatExt, S: SeedExt>(fgn: &Fgn<T, S, Self>, m: usize) -> Array2<T> {
    to_batch(fgn.sample_metal_impl(m), "MetalNative")
  }
}

#[cfg(feature = "accelerate")]
impl Backend for Accelerate {
  fn generate<T: FloatExt, S: SeedExt, S2: SeedExt>(fgn: &Fgn<T, S, Self>, _seed: &S2) -> Array1<T> {
    first_row(fgn.sample_accelerate_impl(1), "Accelerate")
  }
  fn generate_batch<T: FloatExt, S: SeedExt>(fgn: &Fgn<T, S, Self>, m: usize) -> Array2<T> {
    to_batch(fgn.sample_accelerate_impl(m), "Accelerate")
  }
}

#[cfg(any(
  feature = "gpu",
  feature = "cuda-native",
  feature = "cuda-oxide-experimental",
  feature = "accelerate",
  feature = "metal"
))]
fn first_row<T: Clone>(
  result: anyhow::Result<either::Either<Array1<T>, Array2<T>>>,
  name: &str,
) -> Array1<T> {
  match result {
    Ok(either::Either::Left(path)) => path,
    Ok(either::Either::Right(batch)) => batch.row(0).to_owned(),
    Err(e) => panic!("{name} fGN sampling failed: {e}"),
  }
}

#[cfg(any(
  feature = "gpu",
  feature = "cuda-native",
  feature = "cuda-oxide-experimental",
  feature = "accelerate",
  feature = "metal"
))]
fn to_batch<T>(
  result: anyhow::Result<either::Either<Array1<T>, Array2<T>>>,
  name: &str,
) -> Array2<T> {
  match result {
    Ok(either::Either::Left(path)) => path.insert_axis(ndarray::Axis(0)),
    Ok(either::Either::Right(batch)) => batch,
    Err(e) => panic!("{name} fGN sampling failed: {e}"),
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
