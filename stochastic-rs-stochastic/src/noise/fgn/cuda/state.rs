use std::sync::Arc;

use cudarc::cufft;
use cudarc::driver::*;
use cudarc::nvrtc;
use parking_lot::Mutex;

use super::kernels::EXTRACT_F32;
use super::kernels::EXTRACT_F64;
use super::kernels::gen_scale_source;
use crate::device::DeviceError;

type Result<T> = std::result::Result<T, DeviceError>;

pub(super) const CUFFT_FORWARD: i32 = -1;

/// Persistent GPU state: compiled kernels and stream (survives param changes).
#[derive(Clone)]
pub(super) struct GpuKernels {
  pub(super) ordinal: usize,
  pub(super) stream: Arc<CudaStream>,
  pub(super) gen_scale_f32: CudaFunction,
  pub(super) extract_f32: CudaFunction,
  pub(super) gen_scale_f64: CudaFunction,
  pub(super) extract_f64: CudaFunction,
}

/// SAFETY: all GPU ops serialized through the stream.
unsafe impl Send for GpuKernels {}

pub(super) static GPU: Mutex<Option<GpuKernels>> = Mutex::new(None);

/// Cacheable page-locked host buffer, allocated with flags = 0: `alloc_pinned`'s write-combined
/// memory is slow for the host reads done here. Cached per sized context, so pinning happens once.
pub(crate) struct PinnedHost<T> {
  pub(crate) ptr: *mut T,
  pub(super) len: usize,
}

impl<T> PinnedHost<T> {
  /// Elements the allocation holds.
  pub(crate) fn len(&self) -> usize {
    self.len
  }

  /// The first `len` elements, which the caller must have written; `len` is not checked, since
  /// every caller sizes the buffer through [`alloc`](Self::alloc) first.
  ///
  /// # Safety
  ///
  /// `len` must not exceed the allocation, and the copy that filled it must
  /// have completed — a device-to-host copy into page-locked memory is
  /// asynchronous, so the stream has to be synchronised first.
  pub(crate) unsafe fn as_slice(&self, len: usize) -> &[T] {
    unsafe { std::slice::from_raw_parts(self.ptr, len) }
  }

  /// The first `len` elements, to be written into.
  ///
  /// # Safety
  ///
  /// As [`as_slice`](Self::as_slice), minus the completion requirement: this
  /// is the destination a copy is issued against.
  pub(crate) unsafe fn as_mut_slice(&mut self, len: usize) -> &mut [T] {
    unsafe { std::slice::from_raw_parts_mut(self.ptr, len) }
  }

  pub(crate) fn alloc(len: usize) -> Result<Self> {
    let bytes = len * std::mem::size_of::<T>();
    let ptr = unsafe { cudarc::driver::result::malloc_host(bytes, 0) }
      .map_err(|e| DeviceError::Launch(format!("malloc_host: {e}")))? as *mut T;
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

/// A sized context's page-locked staging buffer, allocated by the first transfer that needs it.
pub(super) fn staging<T>(slot: &mut Option<PinnedHost<T>>, len: usize) -> Result<&PinnedHost<T>> {
  if slot.is_none() {
    *slot = Some(PinnedHost::alloc(len)?);
  }
  Ok(slot.as_ref().expect("allocated above"))
}

/// Drops the fGN kernels, cuFFT plans and buffers cached for `ordinal`.
pub(crate) fn release(ordinal: usize) {
  let mut gpu = GPU.lock();
  if frees(gpu.as_ref().map(|k| k.ordinal), ordinal) {
    invalidate(&mut gpu);
  }
}

/// Whether a release for `ordinal` empties a cache of `cached`'s kernels; with none cached it does,
/// since a launch racing an earlier release may have left per-size states behind.
pub(crate) fn frees(cached: Option<usize>, ordinal: usize) -> bool {
  cached.is_none_or(|cached| cached == ordinal)
}

/// Empties the cache: the per-size buffers and plans belong to the context that built them.
fn invalidate(gpu: &mut Option<GpuKernels>) {
  *gpu = None;
  SIZED_F32.lock().clear();
  SIZED_F64.lock().clear();
}

#[cfg(test)]
pub(super) fn cached_sizes() -> usize {
  SIZED_F32.lock().len() + SIZED_F64.lock().len()
}

#[cfg(test)]
pub(super) fn forget_kernels() {
  *GPU.lock() = None;
}

#[cfg(test)]
pub(super) fn cached_context() -> Option<std::sync::Weak<CudaContext>> {
  GPU
    .lock()
    .as_ref()
    .map(|k| Arc::downgrade(k.stream.context()))
}

/// The kernels and stream of the device at `ordinal`, opened if need be and cloned under one lock.
pub(super) fn get_or_init_gpu(ordinal: usize) -> Result<GpuKernels> {
  let mut g = GPU.lock();
  if let Some(k) = g.as_ref().filter(|k| k.ordinal == ordinal) {
    return Ok(k.clone());
  }
  invalidate(&mut g);
  let ctx =
    CudaContext::new(ordinal).map_err(|e| DeviceError::Unavailable(format!("CudaContext: {e}")))?;
  let stream = ctx
    .new_stream()
    .map_err(|e| DeviceError::Launch(format!("stream: {e}")))?;
  let c = stream.context();

  let load = |src: &str, name: &str| -> Result<CudaFunction> {
    let ptx =
      nvrtc::compile_ptx(src).map_err(|e| DeviceError::Compile(format!("NVRTC {name}: {e}")))?;
    let m = c
      .load_module(ptx)
      .map_err(|e| DeviceError::Launch(format!("load {name}: {e}")))?;
    m.load_function(name)
      .map_err(|e| DeviceError::Launch(format!("fn {name}: {e}")))
  };

  let kernels = GpuKernels {
    ordinal,
    gen_scale_f32: load(&gen_scale_source("float"), "gen_scale_float")?,
    extract_f32: load(EXTRACT_F32, "extract_f32")?,
    gen_scale_f64: load(&gen_scale_source("double"), "gen_scale_double")?,
    extract_f64: load(EXTRACT_F64, "extract_f64")?,
    stream,
  };
  *g = Some(kernels.clone());
  Ok(kernels)
}

/// Per-precision GPU state: FFT plan and device buffers, re-created on param change.
pub(super) struct SizedCtxF32 {
  pub(super) fft_plan: cufft::sys::cufftHandle,
  pub(super) d_eigs: CudaSlice<f32>,
  pub(super) d_data: CudaSlice<f32>,
  pub(super) d_out: CudaSlice<f32>,
  pub(super) host_pinned: Option<PinnedHost<f32>>,
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
  pub(super) host_pinned: Option<PinnedHost<f64>>,
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

/// The last [`crate::device::CACHE_SLOTS`] per-size states, least recent first; a launch reuses one
/// only on the stream that built it, which its plan runs on.
pub(super) static SIZED_F32: Mutex<Vec<SizedCtxF32>> = Mutex::new(Vec::new());
pub(super) static SIZED_F64: Mutex<Vec<SizedCtxF64>> = Mutex::new(Vec::new());
