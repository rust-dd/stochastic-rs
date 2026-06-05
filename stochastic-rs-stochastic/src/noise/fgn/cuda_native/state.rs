use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use anyhow::Result;
use cudarc::cufft;
use cudarc::driver::*;
use cudarc::nvrtc;
use parking_lot::Mutex;

use super::kernels::EXTRACT_F32;
use super::kernels::EXTRACT_F64;
use super::kernels::GEN_SCALE_F32;
use super::kernels::GEN_SCALE_F64;

pub(super) const CUFFT_FORWARD: i32 = -1;

/// Persistent GPU state: compiled kernels and stream (survives param changes).
pub(super) struct GpuKernels {
  pub(super) stream: Arc<CudaStream>,
  pub(super) gen_scale_f32: CudaFunction,
  pub(super) extract_f32: CudaFunction,
  pub(super) gen_scale_f64: CudaFunction,
  pub(super) extract_f64: CudaFunction,
}

/// SAFETY: all GPU ops serialized through the stream.
unsafe impl Send for GpuKernels {}

pub(super) static GPU: Mutex<Option<GpuKernels>> = Mutex::new(None);
pub(super) static RNG_SEQ: AtomicU64 = AtomicU64::new(0);

/// Cacheable page-locked host buffer for fast device→host transfers.
///
/// Allocated via the low-level driver with flags = 0 (cacheable). We avoid
/// [`cudarc::driver::CudaContext::alloc_pinned`], which uses
/// `CU_MEMHOSTALLOC_WRITECOMBINED`: write-combined memory is fast for host
/// *writes* but very slow for the host *reads* we do when materialising the
/// output `Array2`. Page-locking lets the driver DMA the result at full PCIe
/// bandwidth instead of staging a pageable copy through a bounce buffer
/// (~3.6 GB/s → ~20 GB/s on PCIe 4.0). Cached in the sized context so the
/// (expensive) pinning happens once per parameter set, not per call.
pub(super) struct PinnedHost<T> {
  pub(super) ptr: *mut T,
  pub(super) len: usize,
}

impl<T> PinnedHost<T> {
  pub(super) fn alloc(len: usize) -> Result<Self> {
    let bytes = len * std::mem::size_of::<T>();
    let ptr = unsafe { cudarc::driver::result::malloc_host(bytes, 0) }
      .map_err(|e| anyhow::anyhow!("malloc_host: {e}"))? as *mut T;
    Ok(Self { ptr, len })
  }
}

impl<T> Drop for PinnedHost<T> {
  fn drop(&mut self) {
    unsafe {
      let _ = cudarc::driver::result::free_host(self.ptr as *mut std::ffi::c_void);
    }
  }
}

unsafe impl<T: Send> Send for PinnedHost<T> {}

pub(super) fn get_or_init_gpu() -> Result<()> {
  let mut g = GPU.lock();
  if g.is_some() {
    return Ok(());
  }
  let ctx = CudaContext::new(0).map_err(|e| anyhow::anyhow!("CudaContext: {e}"))?;
  let stream = ctx
    .new_stream()
    .map_err(|e| anyhow::anyhow!("stream: {e}"))?;
  let c = stream.context();

  let load = |src: &str, name: &str| -> Result<CudaFunction> {
    let ptx = nvrtc::compile_ptx(src).map_err(|e| anyhow::anyhow!("NVRTC {name}: {e}"))?;
    let m = c
      .load_module(ptx)
      .map_err(|e| anyhow::anyhow!("load {name}: {e}"))?;
    m.load_function(name)
      .map_err(|e| anyhow::anyhow!("fn {name}: {e}"))
  };

  *g = Some(GpuKernels {
    gen_scale_f32: load(GEN_SCALE_F32, "gen_scale_f32")?,
    extract_f32: load(EXTRACT_F32, "extract_f32")?,
    gen_scale_f64: load(GEN_SCALE_F64, "gen_scale_f64")?,
    extract_f64: load(EXTRACT_F64, "extract_f64")?,
    stream,
  });
  Ok(())
}

/// Per-precision GPU state: FFT plan and device buffers, re-created on param change.
pub(super) struct SizedCtxF32 {
  pub(super) fft_plan: cufft::sys::cufftHandle,
  pub(super) d_eigs: CudaSlice<f32>,
  pub(super) d_data: CudaSlice<f32>,
  pub(super) d_out: CudaSlice<f32>,
  pub(super) host_pinned: PinnedHost<f32>,
  pub(super) n: usize,
  pub(super) m: usize,
  pub(super) offset: usize,
  pub(super) hurst_bits: u64,
  pub(super) t_bits: u64,
}

impl Drop for SizedCtxF32 {
  fn drop(&mut self) {
    unsafe {
      let _ = cufft::result::destroy(self.fft_plan);
    }
  }
}

pub(super) struct SizedCtxF64 {
  pub(super) fft_plan: cufft::sys::cufftHandle,
  pub(super) d_eigs: CudaSlice<f64>,
  pub(super) d_data: CudaSlice<f64>,
  pub(super) d_out: CudaSlice<f64>,
  pub(super) host_pinned: PinnedHost<f64>,
  pub(super) n: usize,
  pub(super) m: usize,
  pub(super) offset: usize,
  pub(super) hurst_bits: u64,
  pub(super) t_bits: u64,
}

impl Drop for SizedCtxF64 {
  fn drop(&mut self) {
    unsafe {
      let _ = cufft::result::destroy(self.fft_plan);
    }
  }
}

unsafe impl Send for SizedCtxF32 {}
unsafe impl Send for SizedCtxF64 {}

pub(super) static SIZED_F32: Mutex<Option<SizedCtxF32>> = Mutex::new(None);
pub(super) static SIZED_F64: Mutex<Option<SizedCtxF64>> = Mutex::new(None);
