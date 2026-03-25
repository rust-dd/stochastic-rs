//! # cudarc Native CUDA
//!
//! NVIDIA-optimized FGN sampling via cudarc (cuFFT + cuRAND + NVRTC).
//! No `.cu` files, no `nvcc`, no FFI — pure Rust with dynamic CUDA loading.
//!
use std::any::TypeId;
use std::sync::Arc;

use anyhow::Result;
use cudarc::cufft;
use cudarc::curand::CudaRng;
use cudarc::driver::*;
use cudarc::nvrtc;
use either::Either;
use ndarray::Array1;
use ndarray::Array2;
use parking_lot::Mutex;

use super::FGN;
use crate::simd_rng::SeedExt;
use crate::traits::FloatExt;

const CUFFT_FORWARD: i32 = -1;

/// NVRTC kernel: scale interleaved complex data by real eigenvalues (f32).
const SCALE_KERNEL_F32: &str = r#"
extern "C" __global__ void scale_by_eigs_f32(
    float* data,
    const float* sqrt_eigs,
    int traj_size,
    int total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int idx = tid % traj_size;
    float eig = sqrt_eigs[idx];
    data[2*tid]   *= eig;
    data[2*tid+1] *= eig;
}
"#;

const EXTRACT_KERNEL_F32: &str = r#"
extern "C" __global__ void extract_scale_f32(
    const float* data,
    float* output,
    int out_size,
    int traj_stride,
    float scale,
    int total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int traj_id = tid / out_size;
    int idx = tid % out_size;
    int data_idx = traj_id * traj_stride + (idx + 1);
    output[tid] = data[2*data_idx] * scale;
}
"#;

/// NVRTC kernel: scale interleaved complex data by real eigenvalues (f64).
const SCALE_KERNEL_F64: &str = r#"
extern "C" __global__ void scale_by_eigs_f64(
    double* data,
    const double* sqrt_eigs,
    int traj_size,
    int total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int idx = tid % traj_size;
    double eig = sqrt_eigs[idx];
    data[2*tid]   *= eig;
    data[2*tid+1] *= eig;
}
"#;

const EXTRACT_KERNEL_F64: &str = r#"
extern "C" __global__ void extract_scale_f64(
    const double* data,
    double* output,
    int out_size,
    int traj_stride,
    double scale,
    int total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int traj_id = tid / out_size;
    int idx = tid % out_size;
    int data_idx = traj_id * traj_stride + (idx + 1);
    output[tid] = data[2*data_idx] * scale;
}
"#;

struct CudaNativeCtxF32 {
  stream: Arc<CudaStream>,
  scale_fn: CudaFunction,
  extract_fn: CudaFunction,
  fft_plan: cufft::sys::cufftHandle,
  d_sqrt_eigs: CudaSlice<f32>,
  d_data: CudaSlice<f32>,
  d_output: CudaSlice<f32>,
  rng: CudaRng,
  // cache key
  n: usize,
  m: usize,
  offset: usize,
  hurst_bits: u64,
  t_bits: u64,
}

impl Drop for CudaNativeCtxF32 {
  fn drop(&mut self) {
    unsafe {
      let _ = cufft::result::destroy(self.fft_plan);
    }
  }
}

struct CudaNativeCtxF64 {
  stream: Arc<CudaStream>,
  scale_fn: CudaFunction,
  extract_fn: CudaFunction,
  fft_plan: cufft::sys::cufftHandle,
  d_sqrt_eigs: CudaSlice<f64>,
  d_data: CudaSlice<f64>,
  d_output: CudaSlice<f64>,
  rng: CudaRng,
  // cache key
  n: usize,
  m: usize,
  offset: usize,
  hurst_bits: u64,
  t_bits: u64,
}

impl Drop for CudaNativeCtxF64 {
  fn drop(&mut self) {
    unsafe {
      let _ = cufft::result::destroy(self.fft_plan);
    }
  }
}

// SAFETY: The CUDA context and stream are thread-safe.
// All GPU operations are serialized through the stream.
unsafe impl Send for CudaNativeCtxF32 {}
unsafe impl Send for CudaNativeCtxF64 {}

static CTX_F32: Mutex<Option<CudaNativeCtxF32>> = Mutex::new(None);
static CTX_F64: Mutex<Option<CudaNativeCtxF64>> = Mutex::new(None);

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

fn init_ctx_f32(
  stream: &Arc<CudaStream>,
  sqrt_eigs_host: &[f32],
  n: usize,
  m: usize,
  offset: usize,
  hurst_bits: u64,
  t_bits: u64,
) -> Result<CudaNativeCtxF32> {
  let ctx = stream.context();

  // Compile NVRTC kernels
  let ptx_scale = nvrtc::compile_ptx(SCALE_KERNEL_F32)
    .map_err(|e| anyhow::anyhow!("NVRTC compile scale_f32: {e}"))?;
  let ptx_extract = nvrtc::compile_ptx(EXTRACT_KERNEL_F32)
    .map_err(|e| anyhow::anyhow!("NVRTC compile extract_f32: {e}"))?;

  let mod_scale = ctx
    .load_module(ptx_scale)
    .map_err(|e| anyhow::anyhow!("load scale module: {e}"))?;
  let mod_extract = ctx
    .load_module(ptx_extract)
    .map_err(|e| anyhow::anyhow!("load extract module: {e}"))?;

  let scale_fn = mod_scale
    .load_function("scale_by_eigs_f32")
    .map_err(|e| anyhow::anyhow!("load scale_by_eigs_f32: {e}"))?;
  let extract_fn = mod_extract
    .load_function("extract_scale_f32")
    .map_err(|e| anyhow::anyhow!("load extract_scale_f32: {e}"))?;

  // cuFFT batched plan: traj_size complex elements, m batches
  let traj_size = 2 * n;
  let fft_plan =
    cufft::result::plan_1d(traj_size as i32, cufft::sys::cufftType::CUFFT_C2C, m as i32)
      .map_err(|e| anyhow::anyhow!("cuFFT plan_1d: {e}"))?;

  // Device memory
  let d_sqrt_eigs = stream
    .clone_htod(sqrt_eigs_host)
    .map_err(|e| anyhow::anyhow!("htod sqrt_eigs: {e}"))?;
  let d_data = stream
    .alloc_zeros::<f32>(2 * m * traj_size)
    .map_err(|e| anyhow::anyhow!("alloc data: {e}"))?;
  let out_size = n - offset;
  let d_output = stream
    .alloc_zeros::<f32>(m * out_size)
    .map_err(|e| anyhow::anyhow!("alloc output: {e}"))?;

  // cuRAND
  let seed: u64 = rand::Rng::random(&mut crate::simd_rng::rng());
  let rng = CudaRng::new(seed, stream.clone()).map_err(|e| anyhow::anyhow!("cuRAND init: {e}"))?;

  Ok(CudaNativeCtxF32 {
    stream: stream.clone(),
    scale_fn,
    extract_fn,
    fft_plan,
    d_sqrt_eigs,
    d_data,
    d_output,
    rng,
    n,
    m,
    offset,
    hurst_bits,
    t_bits,
  })
}

fn init_ctx_f64(
  stream: &Arc<CudaStream>,
  sqrt_eigs_host: &[f64],
  n: usize,
  m: usize,
  offset: usize,
  hurst_bits: u64,
  t_bits: u64,
) -> Result<CudaNativeCtxF64> {
  let ctx = stream.context();

  let ptx_scale = nvrtc::compile_ptx(SCALE_KERNEL_F64)
    .map_err(|e| anyhow::anyhow!("NVRTC compile scale_f64: {e}"))?;
  let ptx_extract = nvrtc::compile_ptx(EXTRACT_KERNEL_F64)
    .map_err(|e| anyhow::anyhow!("NVRTC compile extract_f64: {e}"))?;

  let mod_scale = ctx
    .load_module(ptx_scale)
    .map_err(|e| anyhow::anyhow!("load scale module: {e}"))?;
  let mod_extract = ctx
    .load_module(ptx_extract)
    .map_err(|e| anyhow::anyhow!("load extract module: {e}"))?;

  let scale_fn = mod_scale
    .load_function("scale_by_eigs_f64")
    .map_err(|e| anyhow::anyhow!("load scale_by_eigs_f64: {e}"))?;
  let extract_fn = mod_extract
    .load_function("extract_scale_f64")
    .map_err(|e| anyhow::anyhow!("load extract_scale_f64: {e}"))?;

  let traj_size = 2 * n;
  let fft_plan =
    cufft::result::plan_1d(traj_size as i32, cufft::sys::cufftType::CUFFT_Z2Z, m as i32)
      .map_err(|e| anyhow::anyhow!("cuFFT plan_1d Z2Z: {e}"))?;

  let d_sqrt_eigs = stream
    .clone_htod(sqrt_eigs_host)
    .map_err(|e| anyhow::anyhow!("htod sqrt_eigs: {e}"))?;
  let d_data = stream
    .alloc_zeros::<f64>(2 * m * traj_size)
    .map_err(|e| anyhow::anyhow!("alloc data: {e}"))?;
  let out_size = n - offset;
  let d_output = stream
    .alloc_zeros::<f64>(m * out_size)
    .map_err(|e| anyhow::anyhow!("alloc output: {e}"))?;

  let seed: u64 = rand::Rng::random(&mut crate::simd_rng::rng());
  let rng = CudaRng::new(seed, stream.clone()).map_err(|e| anyhow::anyhow!("cuRAND init: {e}"))?;

  Ok(CudaNativeCtxF64 {
    stream: stream.clone(),
    scale_fn,
    extract_fn,
    fft_plan,
    d_sqrt_eigs,
    d_data,
    d_output,
    rng,
    n,
    m,
    offset,
    hurst_bits,
    t_bits,
  })
}

fn sample_f32<T: FloatExt>(
  sqrt_eigs: &[f32],
  n: usize,
  m: usize,
  offset: usize,
  hurst: f64,
  t: f64,
) -> Result<Either<Array1<T>, Array2<T>>> {
  let hurst_bits = hurst.to_bits();
  let t_bits = t.to_bits();
  let out_size = n - offset;
  let traj_size = 2 * n;
  let scale_steps = out_size.max(1);
  let scale = (scale_steps as f32).powf(-(hurst as f32)) * (t as f32).powf(hurst as f32);

  let mut guard = CTX_F32.lock();
  let need_init = match &*guard {
    Some(ctx) => {
      ctx.n != n
        || ctx.m != m
        || ctx.offset != offset
        || ctx.hurst_bits != hurst_bits
        || ctx.t_bits != t_bits
    }
    None => true,
  };

  if need_init {
    *guard = None;
    let cuda_ctx = CudaContext::new(0).map_err(|e| anyhow::anyhow!("CudaContext::new: {e}"))?;
    let stream = cuda_ctx
      .new_stream()
      .map_err(|e| anyhow::anyhow!("new_stream: {e}"))?;
    *guard = Some(init_ctx_f32(
      &stream, sqrt_eigs, n, m, offset, hurst_bits, t_bits,
    )?);
  }

  let ctx = guard.as_mut().unwrap();

  // 1. Fill with standard normals (2 * m * traj_size f32s = m * traj_size complex)
  ctx
    .rng
    .fill_with_normal(&mut ctx.d_data, 0.0f32, 1.0f32)
    .map_err(|e| anyhow::anyhow!("cuRAND fill_with_normal: {e}"))?;

  // 2. Scale by eigenvalues
  let total_complex = (m * traj_size) as i32;
  let traj_size_i32 = traj_size as i32;
  unsafe {
    ctx
      .stream
      .launch_builder(&ctx.scale_fn)
      .arg(&mut ctx.d_data)
      .arg(&ctx.d_sqrt_eigs)
      .arg(&traj_size_i32)
      .arg(&total_complex)
      .launch(LaunchConfig::for_num_elems(total_complex as u32))
      .map_err(|e| anyhow::anyhow!("scale kernel launch: {e}"))?;
  }

  // 3. Batched in-place C2C FFT
  {
    let (ptr, _guard) = ctx.d_data.device_ptr_mut(&ctx.stream);
    let complex_ptr = ptr as *mut cufft::sys::cufftComplex;
    unsafe {
      cufft::result::exec_c2c(ctx.fft_plan, complex_ptr, complex_ptr, CUFFT_FORWARD)
        .map_err(|e| anyhow::anyhow!("cuFFT exec_c2c: {e}"))?;
    }
  }

  // 4. Extract real parts + scale
  let total_out = (m * out_size) as i32;
  let out_size_i32 = out_size as i32;
  let traj_stride_i32 = traj_size as i32;
  unsafe {
    ctx
      .stream
      .launch_builder(&ctx.extract_fn)
      .arg(&ctx.d_data)
      .arg(&mut ctx.d_output)
      .arg(&out_size_i32)
      .arg(&traj_stride_i32)
      .arg(&scale)
      .arg(&total_out)
      .launch(LaunchConfig::for_num_elems(total_out as u32))
      .map_err(|e| anyhow::anyhow!("extract kernel launch: {e}"))?;
  }

  // 5. Download results
  let host_output = ctx
    .stream
    .clone_dtoh(&ctx.d_output)
    .map_err(|e| anyhow::anyhow!("dtoh: {e}"))?;

  drop(guard);

  let fgn = array2_from_vec_f32::<T>(host_output, m, out_size);
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
) -> Result<Either<Array1<T>, Array2<T>>> {
  let hurst_bits = hurst.to_bits();
  let t_bits = t.to_bits();
  let out_size = n - offset;
  let traj_size = 2 * n;
  let scale_steps = out_size.max(1);
  let scale = (scale_steps as f64).powf(-hurst) * t.powf(hurst);

  let mut guard = CTX_F64.lock();
  let need_init = match &*guard {
    Some(ctx) => {
      ctx.n != n
        || ctx.m != m
        || ctx.offset != offset
        || ctx.hurst_bits != hurst_bits
        || ctx.t_bits != t_bits
    }
    None => true,
  };

  if need_init {
    *guard = None;
    let cuda_ctx = CudaContext::new(0).map_err(|e| anyhow::anyhow!("CudaContext::new: {e}"))?;
    let stream = cuda_ctx
      .new_stream()
      .map_err(|e| anyhow::anyhow!("new_stream: {e}"))?;
    *guard = Some(init_ctx_f64(
      &stream, sqrt_eigs, n, m, offset, hurst_bits, t_bits,
    )?);
  }

  let ctx = guard.as_mut().unwrap();

  // 1. Fill with standard normals
  ctx
    .rng
    .fill_with_normal(&mut ctx.d_data, 0.0f64, 1.0f64)
    .map_err(|e| anyhow::anyhow!("cuRAND fill_with_normal: {e}"))?;

  // 2. Scale by eigenvalues
  let total_complex = (m * traj_size) as i32;
  let traj_size_i32 = traj_size as i32;
  unsafe {
    ctx
      .stream
      .launch_builder(&ctx.scale_fn)
      .arg(&mut ctx.d_data)
      .arg(&ctx.d_sqrt_eigs)
      .arg(&traj_size_i32)
      .arg(&total_complex)
      .launch(LaunchConfig::for_num_elems(total_complex as u32))
      .map_err(|e| anyhow::anyhow!("scale kernel launch: {e}"))?;
  }

  // 3. Batched in-place Z2Z FFT
  {
    let (ptr, _guard) = ctx.d_data.device_ptr_mut(&ctx.stream);
    let complex_ptr = ptr as *mut cufft::sys::cufftDoubleComplex;
    unsafe {
      cufft::result::exec_z2z(ctx.fft_plan, complex_ptr, complex_ptr, CUFFT_FORWARD)
        .map_err(|e| anyhow::anyhow!("cuFFT exec_z2z: {e}"))?;
    }
  }

  // 4. Extract real parts + scale
  let total_out = (m * out_size) as i32;
  let out_size_i32 = out_size as i32;
  let traj_stride_i32 = traj_size as i32;
  unsafe {
    ctx
      .stream
      .launch_builder(&ctx.extract_fn)
      .arg(&ctx.d_data)
      .arg(&mut ctx.d_output)
      .arg(&out_size_i32)
      .arg(&traj_stride_i32)
      .arg(&scale)
      .arg(&total_out)
      .launch(LaunchConfig::for_num_elems(total_out as u32))
      .map_err(|e| anyhow::anyhow!("extract kernel launch: {e}"))?;
  }

  // 5. Download results
  let host_output = ctx
    .stream
    .clone_dtoh(&ctx.d_output)
    .map_err(|e| anyhow::anyhow!("dtoh: {e}"))?;

  drop(guard);

  let fgn = array2_from_vec_f64::<T>(host_output, m, out_size);
  if m == 1 {
    return Ok(Either::Left(fgn.row(0).to_owned()));
  }
  Ok(Either::Right(fgn))
}

impl<T: FloatExt, S: SeedExt> FGN<T, S> {
  pub(crate) fn sample_cuda_native_impl(&self, m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
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

    // Try f64 first, fall back to f32 on symbol/capability errors
    let eigs_f64: Vec<f64> = self
      .sqrt_eigenvalues
      .iter()
      .map(|x| x.to_f64().unwrap())
      .collect();
    match sample_f64::<T>(&eigs_f64, n, m, offset, hurst, t) {
      Ok(out) => Ok(out),
      Err(_) => {
        let eigs_f32: Vec<f32> = self
          .sqrt_eigenvalues
          .iter()
          .map(|x| x.to_f32().unwrap())
          .collect();
        sample_f32::<T>(&eigs_f32, n, m, offset, hurst, t)
      }
    }
  }
}
