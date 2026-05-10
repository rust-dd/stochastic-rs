//! # cuda-oxide Experimental CUDA
//!
//! Experimental Fgn sampling implemented entirely as NVlabs `cuda-oxide` Rust
//! kernels: Philox/Box-Muller generation, eigenvalue scaling, radix-2 FFT, and
//! output extraction. This is intentionally separate from the stable
//! `cuda-native` backend while the cuda-oxide compiler/runtime are still alpha.

use std::any::TypeId;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use anyhow::Result;
use cuda_core::DeviceBuffer;
use cuda_core::LaunchConfig;
use cuda_device::DisjointSlice;
use cuda_device::kernel;
use cuda_device::thread;
use cuda_host::cuda_launch;
use cuda_host::load_kernel_module;
use either::Either;
use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;

use super::Fgn;
use crate::traits::FloatExt;

const MODULE_ENV: &str = "STOCHASTIC_RS_CUDA_OXIDE_MODULE";
const DEFAULT_MODULE: &str = "stochastic-rs-stochastic";
const TAU_F32: f32 = 6.283_185_5;
const TAU_F64: f64 = 6.283_185_307_179_586;

static RNG_SEQ: AtomicU64 = AtomicU64::new(0);

#[inline]
fn reverse_bits_n(mut x: usize, bits: usize) -> usize {
  let mut y = 0usize;
  let mut i = 0usize;
  while i < bits {
    y = (y << 1) | (x & 1);
    x >>= 1;
    i += 1;
  }
  y
}

#[inline]
fn philox2x32_10(tid: usize, seed: u64, seq: u64) -> (u32, u32) {
  let ctr = tid as u64 + seq;
  let mut lo = ctr as u32;
  let mut hi = (ctr >> 32) as u32;
  let mut k = seed as u32;
  let mut i = 0;
  while i < 10 {
    let p = 0xD251_1F53_u64 * lo as u64;
    lo = ((p >> 32) as u32) ^ hi ^ k;
    hi = p as u32;
    k = k.wrapping_add(0x9E37_79B9);
    i += 1;
  }
  (lo, hi)
}

#[kernel]
pub fn gen_scale_f32(
  mut data: DisjointSlice<f32>,
  sqrt_eigs: &[f32],
  traj_size: usize,
  total_complex: usize,
  seed: u64,
  seq: u64,
) {
  let tid = thread::index_1d().get();
  if tid < total_complex {
    let (lo, hi) = philox2x32_10(tid, seed, seq);
    let u1 = (lo as f32 + 0.5) * 2.328_306_4e-10;
    let u2 = (hi as f32 + 0.5) * 2.328_306_4e-10;
    let r = (-2.0 * u1.ln()).sqrt();
    let angle = TAU_F32 * u2;
    let eig = sqrt_eigs[tid % traj_size];
    let base = 2 * tid;
    unsafe {
      let ptr = data.as_mut_ptr();
      *ptr.add(base) = r * angle.cos() * eig;
      *ptr.add(base + 1) = r * angle.sin() * eig;
    }
  }
}

#[kernel]
pub fn bit_reverse_f32(mut data: DisjointSlice<f32>, traj_size: usize, log_n: usize) {
  let tid = thread::index_1d().get();
  let batch = tid / traj_size;
  let local = tid % traj_size;
  let rev = reverse_bits_n(local, log_n);
  if local < rev {
    let i = 2 * (batch * traj_size + local);
    let j = 2 * (batch * traj_size + rev);
    unsafe {
      let ptr = data.as_mut_ptr();
      let ar = *ptr.add(i);
      let ai = *ptr.add(i + 1);
      *ptr.add(i) = *ptr.add(j);
      *ptr.add(i + 1) = *ptr.add(j + 1);
      *ptr.add(j) = ar;
      *ptr.add(j + 1) = ai;
    }
  }
}

#[kernel]
pub fn fft_stage_f32(mut data: DisjointSlice<f32>, traj_size: usize, half_stride: usize) {
  let tid = thread::index_1d().get();
  let butterflies_per_batch = traj_size / 2;
  let batch = tid / butterflies_per_batch;
  let local = tid % butterflies_per_batch;
  let stride = half_stride * 2;
  let group = local / half_stride;
  let pos = local % half_stride;
  let i_complex = batch * traj_size + group * stride + pos;
  let j_complex = i_complex + half_stride;
  let angle = -TAU_F32 * (pos as f32) / (stride as f32);
  let wr = angle.cos();
  let wi = angle.sin();

  unsafe {
    let ptr = data.as_mut_ptr();
    let i = 2 * i_complex;
    let j = 2 * j_complex;
    let ar = *ptr.add(i);
    let ai = *ptr.add(i + 1);
    let br = *ptr.add(j);
    let bi = *ptr.add(j + 1);
    let tr = br * wr - bi * wi;
    let ti = br * wi + bi * wr;
    *ptr.add(i) = ar + tr;
    *ptr.add(i + 1) = ai + ti;
    *ptr.add(j) = ar - tr;
    *ptr.add(j + 1) = ai - ti;
  }
}

#[kernel]
pub fn extract_real_f32(
  data: &[f32],
  mut output: DisjointSlice<f32>,
  out_size: usize,
  traj_size: usize,
  scale: f32,
) {
  let tid = thread::index_1d();
  if let Some(out) = output.get_mut(tid) {
    let flat = tid.get();
    let traj_id = flat / out_size;
    let idx = flat % out_size;
    *out = data[2 * (traj_id * traj_size + idx + 1)] * scale;
  }
}

#[kernel]
pub fn gen_scale_f64(
  mut data: DisjointSlice<f64>,
  sqrt_eigs: &[f64],
  traj_size: usize,
  total_complex: usize,
  seed: u64,
  seq: u64,
) {
  let tid = thread::index_1d().get();
  if tid < total_complex {
    let (lo, hi) = philox2x32_10(tid, seed, seq);
    let u1 = (lo as f64 + 0.5) * 2.328_306_436_538_696_3e-10;
    let u2 = (hi as f64 + 0.5) * 2.328_306_436_538_696_3e-10;
    let r = (-2.0 * u1.ln()).sqrt();
    let angle = TAU_F64 * u2;
    let eig = sqrt_eigs[tid % traj_size];
    let base = 2 * tid;
    unsafe {
      let ptr = data.as_mut_ptr();
      *ptr.add(base) = r * angle.cos() * eig;
      *ptr.add(base + 1) = r * angle.sin() * eig;
    }
  }
}

#[kernel]
pub fn bit_reverse_f64(mut data: DisjointSlice<f64>, traj_size: usize, log_n: usize) {
  let tid = thread::index_1d().get();
  let batch = tid / traj_size;
  let local = tid % traj_size;
  let rev = reverse_bits_n(local, log_n);
  if local < rev {
    let i = 2 * (batch * traj_size + local);
    let j = 2 * (batch * traj_size + rev);
    unsafe {
      let ptr = data.as_mut_ptr();
      let ar = *ptr.add(i);
      let ai = *ptr.add(i + 1);
      *ptr.add(i) = *ptr.add(j);
      *ptr.add(i + 1) = *ptr.add(j + 1);
      *ptr.add(j) = ar;
      *ptr.add(j + 1) = ai;
    }
  }
}

#[kernel]
pub fn fft_stage_f64(mut data: DisjointSlice<f64>, traj_size: usize, half_stride: usize) {
  let tid = thread::index_1d().get();
  let butterflies_per_batch = traj_size / 2;
  let batch = tid / butterflies_per_batch;
  let local = tid % butterflies_per_batch;
  let stride = half_stride * 2;
  let group = local / half_stride;
  let pos = local % half_stride;
  let i_complex = batch * traj_size + group * stride + pos;
  let j_complex = i_complex + half_stride;
  let angle = -TAU_F64 * (pos as f64) / (stride as f64);
  let wr = angle.cos();
  let wi = angle.sin();

  unsafe {
    let ptr = data.as_mut_ptr();
    let i = 2 * i_complex;
    let j = 2 * j_complex;
    let ar = *ptr.add(i);
    let ai = *ptr.add(i + 1);
    let br = *ptr.add(j);
    let bi = *ptr.add(j + 1);
    let tr = br * wr - bi * wi;
    let ti = br * wi + bi * wr;
    *ptr.add(i) = ar + tr;
    *ptr.add(i + 1) = ai + ti;
    *ptr.add(j) = ar - tr;
    *ptr.add(j + 1) = ai - ti;
  }
}

#[kernel]
pub fn extract_real_f64(
  data: &[f64],
  mut output: DisjointSlice<f64>,
  out_size: usize,
  traj_size: usize,
  scale: f64,
) {
  let tid = thread::index_1d();
  if let Some(out) = output.get_mut(tid) {
    let flat = tid.get();
    let traj_id = flat / out_size;
    let idx = flat % out_size;
    *out = data[2 * (traj_id * traj_size + idx + 1)] * scale;
  }
}

fn module_name() -> String {
  if let Ok(name) = std::env::var(MODULE_ENV) {
    return name;
  }
  std::env::current_exe()
    .ok()
    .and_then(|path| {
      path
        .file_stem()
        .map(|name| name.to_string_lossy().into_owned())
    })
    .unwrap_or_else(|| DEFAULT_MODULE.to_string())
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
  module_name: &str,
) -> Result<Either<Array1<T>, Array2<T>>> {
  let out_size = n - offset;
  let traj_size = 2 * n;
  let total_complex = m * traj_size;
  let scale = (out_size.max(1) as f32).powf(-(hurst as f32)) * (t as f32).powf(hurst as f32);

  let ctx = cuda_core::CudaContext::new(0).map_err(|e| anyhow::anyhow!("CudaContext: {e}"))?;
  let stream = ctx.default_stream();
  let module = load_kernel_module(&ctx, module_name)
    .map_err(|e| anyhow::anyhow!("cuda-oxide module `{module_name}`: {e}"))?;

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
  if m == 1 {
    return Ok(Either::Left(fgn.row(0).to_owned()));
  }
  Ok(Either::Right(fgn))
}

fn sample_f64<T: FloatExt>(
  sqrt_eigs: &[f64],
  n: usize,
  m: usize,
  offset: usize,
  hurst: f64,
  t: f64,
  module_name: &str,
) -> Result<Either<Array1<T>, Array2<T>>> {
  let out_size = n - offset;
  let traj_size = 2 * n;
  let total_complex = m * traj_size;
  let scale = (out_size.max(1) as f64).powf(-hurst) * t.powf(hurst);

  let ctx = cuda_core::CudaContext::new(0).map_err(|e| anyhow::anyhow!("CudaContext: {e}"))?;
  let stream = ctx.default_stream();
  let module = load_kernel_module(&ctx, module_name)
    .map_err(|e| anyhow::anyhow!("cuda-oxide module `{module_name}`: {e}"))?;

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
  if m == 1 {
    return Ok(Either::Left(fgn.row(0).to_owned()));
  }
  Ok(Either::Right(fgn))
}

impl<T: FloatExt, S: SeedExt> Fgn<T, S> {
  /// Sample with the experimental cuda-oxide backend using a specific module stem.
  ///
  /// The module stem must match the `.ptx`/`.ll` artifact generated by
  /// `cargo oxide build`. For downstream binaries this is normally the package
  /// or binary name; `sample_cuda_oxide` uses `STOCHASTIC_RS_CUDA_OXIDE_MODULE`
  /// when set and falls back to `stochastic-rs-stochastic`.
  pub fn sample_cuda_oxide_with_module(
    &self,
    m: usize,
    module_name: &str,
  ) -> Result<Either<Array1<T>, Array2<T>>> {
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
      return sample_f32::<T>(&eigs, n, m, offset, hurst, t, module_name);
    }

    let eigs: Vec<f64> = self
      .sqrt_eigenvalues
      .iter()
      .map(|x| x.to_f64().unwrap())
      .collect();
    sample_f64::<T>(&eigs, n, m, offset, hurst, t, module_name)
  }

  pub(crate) fn sample_cuda_oxide_impl(&self, m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    self.sample_cuda_oxide_with_module(m, &module_name())
  }
}
