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

use crate::noise::fgn::Fgn;
use crate::traits::FloatExt;

/// CPU backend — the default `B` for every process.
pub struct Cpu;

/// cudarc + cuFFT + NVRTC Philox.
#[cfg(feature = "cuda-native")]
pub struct CudaNative;

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

  /// `m` fGN paths in one batched call, one [`Array1`] per path.
  fn generate_batch<T: FloatExt, S: SeedExt>(fgn: &Fgn<T, S, Self>, m: usize) -> Vec<Array1<T>>;

  /// Two independent fGN paths in one pass. Default: a batch of two; [`Cpu`]
  /// overrides with the real/imag parts of a single circulant FFT (one FFT,
  /// two independent fields — Dietrich & Newsam). The host-side `seed` drives
  /// the CPU path only.
  fn generate_pair<T: FloatExt, S: SeedExt, S2: SeedExt>(
    fgn: &Fgn<T, S, Self>,
    seed: &S2,
  ) -> (Array1<T>, Array1<T>) {
    let _ = seed;
    let mut paths = Self::generate_batch(fgn, 2);
    let second = paths.pop().expect("generate_batch(2) yields two paths");
    let first = paths.pop().expect("generate_batch(2) yields two paths");
    (first, second)
  }
}

impl Backend for Cpu {
  fn generate<T: FloatExt, S: SeedExt, S2: SeedExt>(fgn: &Fgn<T, S, Self>, seed: &S2) -> Array1<T> {
    fgn.sample_cpu_impl(seed)
  }

  fn generate_batch<T: FloatExt, S: SeedExt>(fgn: &Fgn<T, S, Self>, m: usize) -> Vec<Array1<T>> {
    (0..m).into_par_iter().map(|_| fgn.sample_cpu()).collect()
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
/// host-side seed is unused (GPU backends carry their own RNG). Each marker and
/// its impl are gated on the backend's feature.
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

      fn generate_batch<T: FloatExt, S: SeedExt>(
        fgn: &Fgn<T, S, Self>,
        m: usize,
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

/// Accelerate (vDSP) runs on the CPU, so the batch is parallelised across cores
/// at the device level — one fast single-path FFT per task, like [`Cpu`] — not
/// looped on-device like the true GPU backends. Each worker reuses its own
/// thread-local vDSP setup and scratch.
#[cfg(feature = "accelerate")]
impl Backend for Accelerate {
  fn generate<T: FloatExt, S: SeedExt, S2: SeedExt>(
    fgn: &Fgn<T, S, Self>,
    _seed: &S2,
  ) -> Array1<T> {
    fgn.sample_accelerate_impl(1).unwrap().row(0).to_owned()
  }

  fn generate_batch<T: FloatExt, S: SeedExt>(fgn: &Fgn<T, S, Self>, m: usize) -> Vec<Array1<T>> {
    (0..m)
      .into_par_iter()
      .map(|_| fgn.sample_accelerate_impl(1).unwrap().row(0).to_owned())
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
