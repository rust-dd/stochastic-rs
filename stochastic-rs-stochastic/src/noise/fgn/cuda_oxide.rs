//! # cuda-oxide Experimental CUDA
//!
//! Experimental Fgn sampling whose device kernels are written in pure Rust and
//! compiled to NVVM IR by the NVlabs `cuda-oxide` rustc codegen backend. The
//! kernels live in the device-only `fgn-oxide-kernels` crate; their NVVM IR is
//! generated once (by the maintainer, via `cargo oxide`) and committed here as
//! `fgn_oxide_kernels.ll`. It is `include_bytes!`-embedded and lowered to a
//! cubin at runtime with libNVVM + nvJitLink, so a plain downstream
//! `cargo build` runs with no `cargo oxide` / precompile step.
//!
//! This backend is intentionally separate from the stable `cuda-native`
//! backend while the cuda-oxide compiler/runtime are still alpha.

use std::any::TypeId;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use anyhow::Result;
use cuda_core::DeviceBuffer;
use cuda_core::LaunchConfig;
use cuda_host::cuda_launch;
use fgn_oxide_kernels::*;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;

use super::Fgn;
use crate::traits::FloatExt;

/// NVVM IR for the fGN device kernels, generated once with the cuda-oxide
/// rustc codegen backend (`fgn-oxide-kernels` crate) and committed. Embedded so
/// downstream needs no `cargo oxide`; lowered to a cubin on first use below.
const FGN_OXIDE_NVVM_IR: &[u8] = include_bytes!("fgn_oxide_kernels.ll");

static RNG_SEQ: AtomicU64 = AtomicU64::new(0);

/// Path to the cubin lowered from [`FGN_OXIDE_NVVM_IR`], cached for the process.
/// Compilation is deterministic per GPU arch, so one compile serves every call.
static FGN_OXIDE_CUBIN_PATH: OnceLock<std::path::PathBuf> = OnceLock::new();

/// Lower the embedded NVVM IR to a cubin (libNVVM + nvJitLink, linking
/// `libdevice` for the `ln`/`sqrt`/`cos`/`sin` the RNG uses) and return its
/// path, cached after the first call. The IR is written to a temp file because
/// the runtime's `build_cubin_from_ll` is path-based. Needs the CUDA Toolkit's
/// libNVVM at runtime — but no `cargo oxide` / precompile step downstream.
fn fgn_oxide_module_path(arch: &str) -> Result<&'static str> {
  if let Some(path) = FGN_OXIDE_CUBIN_PATH.get() {
    return Ok(path.to_str().expect("cubin path is valid UTF-8"));
  }
  let dir = std::env::temp_dir().join("stochastic_rs_cuda_oxide");
  std::fs::create_dir_all(&dir).map_err(|e| anyhow::anyhow!("cuda-oxide tmp dir: {e}"))?;
  let ll_path = dir.join("fgn_oxide_kernels.ll");
  std::fs::write(&ll_path, FGN_OXIDE_NVVM_IR)
    .map_err(|e| anyhow::anyhow!("cuda-oxide write IR: {e}"))?;
  let cubin = cuda_host::ltoir::build_cubin_from_ll(&ll_path, arch)
    .map_err(|e| anyhow::anyhow!("cuda-oxide: build cubin from embedded NVVM IR: {e}"))?;
  let _ = FGN_OXIDE_CUBIN_PATH.set(cubin);
  Ok(
    FGN_OXIDE_CUBIN_PATH
      .get()
      .expect("just set")
      .to_str()
      .expect("cubin path is valid UTF-8"),
  )
}

fn array2_from_flat<T: FloatExt, U: Copy + Into<f64>>(
  host: &[U],
  m: usize,
  cols: usize,
) -> Array2<T> {
  let mut out = Array2::<T>::zeros((m, cols));
  for i in 0..m {
    for j in 0..cols {
      out[[i, j]] = T::from_f64_fast(host[i * cols + j].into());
    }
  }
  out
}

fn array2_from_vec_f32<T: FloatExt>(v: Vec<f32>, m: usize, cols: usize) -> Array2<T> {
  if TypeId::of::<T>() == TypeId::of::<f32>() {
    let out = Array2::<f32>::from_shape_vec((m, cols), v).expect("shape must be valid");
    unsafe { std::mem::transmute::<Array2<f32>, Array2<T>>(out) }
  } else {
    array2_from_flat::<T, f32>(&v, m, cols)
  }
}

fn array2_from_vec_f64<T: FloatExt>(v: Vec<f64>, m: usize, cols: usize) -> Array2<T> {
  if TypeId::of::<T>() == TypeId::of::<f64>() {
    let out = Array2::<f64>::from_shape_vec((m, cols), v).expect("shape must be valid");
    unsafe { std::mem::transmute::<Array2<f64>, Array2<T>>(out) }
  } else {
    array2_from_flat::<T, f64>(&v, m, cols)
  }
}

fn sample_f32<T: FloatExt>(
  sqrt_eigs: &[f32],
  n: usize,
  m: usize,
  offset: usize,
  hurst: f64,
  t: f64,
) -> Result<Array2<T>> {
  let out_size = n - offset;
  let traj_size = 2 * n;
  let total_complex = m * traj_size;
  let scale = (out_size.max(1) as f32).powf(-(hurst as f32)) * (t as f32).powf(hurst as f32);

  let ctx = cuda_core::CudaContext::new(0).map_err(|e| anyhow::anyhow!("CudaContext: {e}"))?;
  let stream = ctx.default_stream();
  let (cc_major, cc_minor) = ctx
    .compute_capability()
    .map_err(|e| anyhow::anyhow!("cuda-oxide compute capability: {e}"))?;
  let arch = format!("sm_{cc_major}{cc_minor}");
  let module = ctx
    .load_module_from_file(fgn_oxide_module_path(&arch)?)
    .map_err(|e| anyhow::anyhow!("cuda-oxide load module: {e}"))?;

  let mut data_dev = DeviceBuffer::<f32>::zeroed(&stream, 2 * total_complex)
    .map_err(|e| anyhow::anyhow!("cuda-oxide alloc complex data: {e}"))?;
  let eigs_dev = DeviceBuffer::from_host(&stream, sqrt_eigs)
    .map_err(|e| anyhow::anyhow!("cuda-oxide htod eigs: {e}"))?;
  let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, m * out_size)
    .map_err(|e| anyhow::anyhow!("cuda-oxide alloc out: {e}"))?;

  let seed: u64 = rand::Rng::random(&mut crate::simd_rng::rng());
  let seq = RNG_SEQ.fetch_add(total_complex as u64, Ordering::Relaxed);
  cuda_launch! {
    kernel: gen_scale_f32,
    stream: stream,
    module: module,
    config: LaunchConfig::for_num_elems(total_complex as u32),
    args: [slice_mut(data_dev), slice(eigs_dev), traj_size, total_complex, seed, seq]
  }
  .map_err(|e| anyhow::anyhow!("cuda-oxide gen_scale_f32: {e}"))?;

  let log_n = traj_size.trailing_zeros() as usize;
  cuda_launch! {
    kernel: bit_reverse_f32,
    stream: stream,
    module: module,
    config: LaunchConfig::for_num_elems(total_complex as u32),
    args: [slice_mut(data_dev), traj_size, log_n]
  }
  .map_err(|e| anyhow::anyhow!("cuda-oxide bit_reverse_f32: {e}"))?;

  let total_butterflies = m * (traj_size / 2);
  let mut half_stride = 1usize;
  while half_stride < traj_size {
    cuda_launch! {
      kernel: fft_stage_f32,
      stream: stream,
      module: module,
      config: LaunchConfig::for_num_elems(total_butterflies as u32),
      args: [slice_mut(data_dev), traj_size, half_stride]
    }
    .map_err(|e| anyhow::anyhow!("cuda-oxide fft_stage_f32({half_stride}): {e}"))?;
    half_stride *= 2;
  }

  cuda_launch! {
    kernel: extract_real_f32,
    stream: stream,
    module: module,
    config: LaunchConfig::for_num_elems((m * out_size) as u32),
    args: [slice(data_dev), slice_mut(out_dev), out_size, traj_size, scale]
  }
  .map_err(|e| anyhow::anyhow!("cuda-oxide extract_real_f32: {e}"))?;

  let host = out_dev
    .to_host_vec(&stream)
    .map_err(|e| anyhow::anyhow!("cuda-oxide dtoh: {e}"))?;
  let fgn = array2_from_vec_f32::<T>(host, m, out_size);
  Ok(fgn)
}

fn sample_f64<T: FloatExt>(
  sqrt_eigs: &[f64],
  n: usize,
  m: usize,
  offset: usize,
  hurst: f64,
  t: f64,
) -> Result<Array2<T>> {
  let out_size = n - offset;
  let traj_size = 2 * n;
  let total_complex = m * traj_size;
  let scale = (out_size.max(1) as f64).powf(-hurst) * t.powf(hurst);

  let ctx = cuda_core::CudaContext::new(0).map_err(|e| anyhow::anyhow!("CudaContext: {e}"))?;
  let stream = ctx.default_stream();
  let (cc_major, cc_minor) = ctx
    .compute_capability()
    .map_err(|e| anyhow::anyhow!("cuda-oxide compute capability: {e}"))?;
  let arch = format!("sm_{cc_major}{cc_minor}");
  let module = ctx
    .load_module_from_file(fgn_oxide_module_path(&arch)?)
    .map_err(|e| anyhow::anyhow!("cuda-oxide load module: {e}"))?;

  let mut data_dev = DeviceBuffer::<f64>::zeroed(&stream, 2 * total_complex)
    .map_err(|e| anyhow::anyhow!("cuda-oxide alloc complex data: {e}"))?;
  let eigs_dev = DeviceBuffer::from_host(&stream, sqrt_eigs)
    .map_err(|e| anyhow::anyhow!("cuda-oxide htod eigs: {e}"))?;
  let mut out_dev = DeviceBuffer::<f64>::zeroed(&stream, m * out_size)
    .map_err(|e| anyhow::anyhow!("cuda-oxide alloc out: {e}"))?;

  let seed: u64 = rand::Rng::random(&mut crate::simd_rng::rng());
  let seq = RNG_SEQ.fetch_add(total_complex as u64, Ordering::Relaxed);
  cuda_launch! {
    kernel: gen_scale_f64,
    stream: stream,
    module: module,
    config: LaunchConfig::for_num_elems(total_complex as u32),
    args: [slice_mut(data_dev), slice(eigs_dev), traj_size, total_complex, seed, seq]
  }
  .map_err(|e| anyhow::anyhow!("cuda-oxide gen_scale_f64: {e}"))?;

  let log_n = traj_size.trailing_zeros() as usize;
  cuda_launch! {
    kernel: bit_reverse_f64,
    stream: stream,
    module: module,
    config: LaunchConfig::for_num_elems(total_complex as u32),
    args: [slice_mut(data_dev), traj_size, log_n]
  }
  .map_err(|e| anyhow::anyhow!("cuda-oxide bit_reverse_f64: {e}"))?;

  let total_butterflies = m * (traj_size / 2);
  let mut half_stride = 1usize;
  while half_stride < traj_size {
    cuda_launch! {
      kernel: fft_stage_f64,
      stream: stream,
      module: module,
      config: LaunchConfig::for_num_elems(total_butterflies as u32),
      args: [slice_mut(data_dev), traj_size, half_stride]
    }
    .map_err(|e| anyhow::anyhow!("cuda-oxide fft_stage_f64({half_stride}): {e}"))?;
    half_stride *= 2;
  }

  cuda_launch! {
    kernel: extract_real_f64,
    stream: stream,
    module: module,
    config: LaunchConfig::for_num_elems((m * out_size) as u32),
    args: [slice(data_dev), slice_mut(out_dev), out_size, traj_size, scale]
  }
  .map_err(|e| anyhow::anyhow!("cuda-oxide extract_real_f64: {e}"))?;

  let host = out_dev
    .to_host_vec(&stream)
    .map_err(|e| anyhow::anyhow!("cuda-oxide dtoh: {e}"))?;
  let fgn = array2_from_vec_f64::<T>(host, m, out_size);
  Ok(fgn)
}

impl<T: FloatExt, S: SeedExt, B> Fgn<T, S, B> {
  /// Sample with the experimental cuda-oxide backend.
  ///
  /// Device kernels are embedded as NVVM IR and lowered to a cubin at runtime
  /// (libNVVM + nvJitLink), so no `cargo oxide` precompile step is needed
  /// downstream. `_module_name` is accepted for backwards compatibility but
  /// ignored — the embedded module is always used.
  pub fn sample_cuda_oxide_with_module(&self, m: usize, _module_name: &str) -> Result<Array2<T>> {
    let n = self.n;
    let offset = self.offset;
    let hurst = self.hurst.to_f64().unwrap();
    let t = self.t.unwrap_or(T::one()).to_f64().unwrap();

    if TypeId::of::<T>() == TypeId::of::<f32>() {
      let eigs: Vec<f32> = self
        .sqrt_eigenvalues
        .iter()
        .map(|x| x.to_f32().unwrap())
        .collect();
      return sample_f32::<T>(&eigs, n, m, offset, hurst, t);
    }

    let eigs: Vec<f64> = self
      .sqrt_eigenvalues
      .iter()
      .map(|x| x.to_f64().unwrap())
      .collect();
    sample_f64::<T>(&eigs, n, m, offset, hurst, t)
  }

  pub(crate) fn sample_cuda_oxide_impl(&self, m: usize) -> Result<Array2<T>> {
    self.sample_cuda_oxide_with_module(m, "")
  }
}
