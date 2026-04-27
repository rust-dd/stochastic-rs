//! # cudarc Native CUDA
//!
//! NVIDIA-optimized Fgn sampling via cudarc (cuFFT + NVRTC).
//! Fused Philox RNG + eigenvalue scaling eliminates cuRAND dependency
//! and one GPU memory round-trip.
//!
use std::any::TypeId;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use anyhow::Result;
use cudarc::cufft;
use cudarc::driver::*;
use cudarc::nvrtc;
use either::Either;
use ndarray::Array1;
use ndarray::Array2;
use parking_lot::Mutex;
use stochastic_rs_core::simd_rng::SeedExt;

use super::Fgn;
use crate::traits::FloatExt;

const CUFFT_FORWARD: i32 = -1;

/// Fused generate + scale NVRTC kernel (Philox-2x32-10 RNG + Box-Muller, f32).
/// Generates two standard normals per thread, scales by sqrt_eigenvalue,
/// and writes interleaved complex output.
const GEN_SCALE_F32: &str = r#"
extern "C" __global__ void gen_scale_f32(
    float* __restrict__ data,
    const float* __restrict__ sqrt_eigs,
    int traj_size, int total,
    unsigned long long seed, unsigned long long seq)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    /* Philox-2x32-10 */
    unsigned int lo = (unsigned int)((unsigned long long)tid + seq);
    unsigned int hi = (unsigned int)(((unsigned long long)tid + seq) >> 32);
    unsigned int k  = (unsigned int)seed;
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        unsigned long long p = (unsigned long long)0xD2511F53u * lo;
        lo = ((unsigned int)(p >> 32)) ^ hi ^ k;
        hi = (unsigned int)p;
        k += 0x9E3779B9u;
    }

    float u1 = (lo + 0.5f) * 2.3283064365386963e-10f;
    float u2 = (hi + 0.5f) * 2.3283064365386963e-10f;
    float r  = sqrtf(-2.0f * logf(u1));
    float sn, cs;
    __sincosf(6.283185307179586f * u2, &sn, &cs);

    float eig = sqrt_eigs[tid % traj_size];
    data[2*tid]   = r * cs * eig;
    data[2*tid+1] = r * sn * eig;
}
"#;

const EXTRACT_F32: &str = r#"
extern "C" __global__ void extract_f32(
    const float* __restrict__ data,
    float* __restrict__ output,
    int out_size, int traj_stride, float scale, int total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int traj_id  = tid / out_size;
    int idx      = tid % out_size;
    output[tid]  = data[2 * (traj_id * traj_stride + idx + 1)] * scale;
}
"#;

const GEN_SCALE_F64: &str = r#"
extern "C" __global__ void gen_scale_f64(
    double* __restrict__ data,
    const double* __restrict__ sqrt_eigs,
    int traj_size, int total,
    unsigned long long seed, unsigned long long seq)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    unsigned int lo = (unsigned int)((unsigned long long)tid + seq);
    unsigned int hi = (unsigned int)(((unsigned long long)tid + seq) >> 32);
    unsigned int k  = (unsigned int)seed;
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        unsigned long long p = (unsigned long long)0xD2511F53u * lo;
        lo = ((unsigned int)(p >> 32)) ^ hi ^ k;
        hi = (unsigned int)p;
        k += 0x9E3779B9u;
    }

    double u1 = ((double)lo + 0.5) * 2.3283064365386963e-10;
    double u2 = ((double)hi + 0.5) * 2.3283064365386963e-10;
    double r  = sqrt(-2.0 * log(u1));
    double angle = 6.283185307179586 * u2;

    double eig = sqrt_eigs[tid % traj_size];
    data[2*tid]   = r * cos(angle) * eig;
    data[2*tid+1] = r * sin(angle) * eig;
}
"#;

const EXTRACT_F64: &str = r#"
extern "C" __global__ void extract_f64(
    const double* __restrict__ data,
    double* __restrict__ output,
    int out_size, int traj_stride, double scale, int total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int traj_id  = tid / out_size;
    int idx      = tid % out_size;
    output[tid]  = data[2 * (traj_id * traj_stride + idx + 1)] * scale;
}
"#;

/// Persistent GPU state: compiled kernels and stream (survives param changes).
struct GpuKernels {
  stream: Arc<CudaStream>,
  gen_scale_f32: CudaFunction,
  extract_f32: CudaFunction,
  gen_scale_f64: CudaFunction,
  extract_f64: CudaFunction,
}

/// SAFETY: all GPU ops serialized through the stream.
unsafe impl Send for GpuKernels {}

static GPU: Mutex<Option<GpuKernels>> = Mutex::new(None);
static RNG_SEQ: AtomicU64 = AtomicU64::new(0);

fn get_or_init_gpu() -> Result<()> {
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
struct SizedCtxF32 {
  fft_plan: cufft::sys::cufftHandle,
  d_eigs: CudaSlice<f32>,
  d_data: CudaSlice<f32>,
  d_out: CudaSlice<f32>,
  n: usize,
  m: usize,
  offset: usize,
  hurst_bits: u64,
  t_bits: u64,
}

impl Drop for SizedCtxF32 {
  fn drop(&mut self) {
    unsafe {
      let _ = cufft::result::destroy(self.fft_plan);
    }
  }
}

struct SizedCtxF64 {
  fft_plan: cufft::sys::cufftHandle,
  d_eigs: CudaSlice<f64>,
  d_data: CudaSlice<f64>,
  d_out: CudaSlice<f64>,
  n: usize,
  m: usize,
  offset: usize,
  hurst_bits: u64,
  t_bits: u64,
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

static SIZED_F32: Mutex<Option<SizedCtxF32>> = Mutex::new(None);
static SIZED_F64: Mutex<Option<SizedCtxF64>> = Mutex::new(None);

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
) -> Result<Either<Array1<T>, Array2<T>>> {
  let hurst_bits = hurst.to_bits();
  let t_bits = t.to_bits();
  let out_size = n - offset;
  let traj_size = 2 * n;
  let scale = (out_size.max(1) as f32).powf(-(hurst as f32)) * (t as f32).powf(hurst as f32);

  get_or_init_gpu()?;
  let gpu = GPU.lock();
  let gpu = gpu.as_ref().unwrap();

  let mut sized = SIZED_F32.lock();
  let need_init = match &*sized {
    Some(s) => {
      s.n != n || s.m != m || s.offset != offset || s.hurst_bits != hurst_bits || s.t_bits != t_bits
    }
    None => true,
  };

  if need_init {
    *sized = None;
    let plan = cufft::result::plan_1d(traj_size as i32, cufft::sys::cufftType::CUFFT_C2C, m as i32)
      .map_err(|e| anyhow::anyhow!("cuFFT plan: {e}"))?;
    unsafe {
      cufft::result::set_stream(plan, gpu.stream.cu_stream() as _)
        .map_err(|e| anyhow::anyhow!("cuFFT set_stream: {e}"))?;
    }
    *sized = Some(SizedCtxF32 {
      fft_plan: plan,
      d_eigs: gpu
        .stream
        .clone_htod(sqrt_eigs)
        .map_err(|e| anyhow::anyhow!("htod eigs: {e}"))?,
      d_data: gpu
        .stream
        .alloc_zeros::<f32>(2 * m * traj_size)
        .map_err(|e| anyhow::anyhow!("alloc data: {e}"))?,
      d_out: gpu
        .stream
        .alloc_zeros::<f32>(m * out_size)
        .map_err(|e| anyhow::anyhow!("alloc out: {e}"))?,
      n,
      m,
      offset,
      hurst_bits,
      t_bits,
    });
  }

  let s = sized.as_mut().unwrap();

  // 1. Fused generate normals + scale by eigenvalues
  let total_complex = (m * traj_size) as i32;
  let traj_i32 = traj_size as i32;
  let seed: u64 = rand::Rng::random(&mut crate::simd_rng::rng());
  let seq = RNG_SEQ.fetch_add(total_complex as u64, Ordering::Relaxed);
  unsafe {
    gpu
      .stream
      .launch_builder(&gpu.gen_scale_f32)
      .arg(&mut s.d_data)
      .arg(&s.d_eigs)
      .arg(&traj_i32)
      .arg(&total_complex)
      .arg(&seed)
      .arg(&seq)
      .launch(LaunchConfig::for_num_elems(total_complex as u32))
      .map_err(|e| anyhow::anyhow!("gen_scale: {e}"))?;
  }

  // 2. Batched FFT
  {
    let (ptr, _g) = s.d_data.device_ptr_mut(&gpu.stream);
    unsafe {
      cufft::result::exec_c2c(s.fft_plan, ptr as *mut _, ptr as *mut _, CUFFT_FORWARD)
        .map_err(|e| anyhow::anyhow!("cuFFT: {e}"))?;
    }
  }

  // 3. Extract real parts + scale
  let total_out = (m * out_size) as i32;
  let out_i32 = out_size as i32;
  let stride_i32 = traj_size as i32;
  unsafe {
    gpu
      .stream
      .launch_builder(&gpu.extract_f32)
      .arg(&s.d_data)
      .arg(&mut s.d_out)
      .arg(&out_i32)
      .arg(&stride_i32)
      .arg(&scale)
      .arg(&total_out)
      .launch(LaunchConfig::for_num_elems(total_out as u32))
      .map_err(|e| anyhow::anyhow!("extract: {e}"))?;
  }

  // 4. DtoH
  let host = gpu
    .stream
    .clone_dtoh(&s.d_out)
    .map_err(|e| anyhow::anyhow!("dtoh: {e}"))?;
  drop(sized);

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
) -> Result<Either<Array1<T>, Array2<T>>> {
  let hurst_bits = hurst.to_bits();
  let t_bits = t.to_bits();
  let out_size = n - offset;
  let traj_size = 2 * n;
  let scale = (out_size.max(1) as f64).powf(-hurst) * t.powf(hurst);

  get_or_init_gpu()?;
  let gpu = GPU.lock();
  let gpu = gpu.as_ref().unwrap();

  let mut sized = SIZED_F64.lock();
  let need_init = match &*sized {
    Some(s) => {
      s.n != n || s.m != m || s.offset != offset || s.hurst_bits != hurst_bits || s.t_bits != t_bits
    }
    None => true,
  };

  if need_init {
    *sized = None;
    let plan = cufft::result::plan_1d(traj_size as i32, cufft::sys::cufftType::CUFFT_Z2Z, m as i32)
      .map_err(|e| anyhow::anyhow!("cuFFT plan: {e}"))?;
    unsafe {
      cufft::result::set_stream(plan, gpu.stream.cu_stream() as _)
        .map_err(|e| anyhow::anyhow!("cuFFT set_stream: {e}"))?;
    }
    *sized = Some(SizedCtxF64 {
      fft_plan: plan,
      d_eigs: gpu
        .stream
        .clone_htod(sqrt_eigs)
        .map_err(|e| anyhow::anyhow!("htod eigs: {e}"))?,
      d_data: gpu
        .stream
        .alloc_zeros::<f64>(2 * m * traj_size)
        .map_err(|e| anyhow::anyhow!("alloc data: {e}"))?,
      d_out: gpu
        .stream
        .alloc_zeros::<f64>(m * out_size)
        .map_err(|e| anyhow::anyhow!("alloc out: {e}"))?,
      n,
      m,
      offset,
      hurst_bits,
      t_bits,
    });
  }

  let s = sized.as_mut().unwrap();

  // 1. Fused generate + scale
  let total_complex = (m * traj_size) as i32;
  let traj_i32 = traj_size as i32;
  let seed: u64 = rand::Rng::random(&mut crate::simd_rng::rng());
  let seq = RNG_SEQ.fetch_add(total_complex as u64, Ordering::Relaxed);
  unsafe {
    gpu
      .stream
      .launch_builder(&gpu.gen_scale_f64)
      .arg(&mut s.d_data)
      .arg(&s.d_eigs)
      .arg(&traj_i32)
      .arg(&total_complex)
      .arg(&seed)
      .arg(&seq)
      .launch(LaunchConfig::for_num_elems(total_complex as u32))
      .map_err(|e| anyhow::anyhow!("gen_scale: {e}"))?;
  }

  // 2. Batched FFT
  {
    let (ptr, _g) = s.d_data.device_ptr_mut(&gpu.stream);
    unsafe {
      cufft::result::exec_z2z(s.fft_plan, ptr as *mut _, ptr as *mut _, CUFFT_FORWARD)
        .map_err(|e| anyhow::anyhow!("cuFFT: {e}"))?;
    }
  }

  // 3. Extract + scale
  let total_out = (m * out_size) as i32;
  let out_i32 = out_size as i32;
  let stride_i32 = traj_size as i32;
  unsafe {
    gpu
      .stream
      .launch_builder(&gpu.extract_f64)
      .arg(&s.d_data)
      .arg(&mut s.d_out)
      .arg(&out_i32)
      .arg(&stride_i32)
      .arg(&scale)
      .arg(&total_out)
      .launch(LaunchConfig::for_num_elems(total_out as u32))
      .map_err(|e| anyhow::anyhow!("extract: {e}"))?;
  }

  // 4. DtoH
  let host = gpu
    .stream
    .clone_dtoh(&s.d_out)
    .map_err(|e| anyhow::anyhow!("dtoh: {e}"))?;
  drop(sized);

  let fgn = array2_from_vec_f64::<T>(host, m, out_size);
  if m == 1 {
    return Ok(Either::Left(fgn.row(0).to_owned()));
  }
  Ok(Either::Right(fgn))
}

impl<T: FloatExt, S: SeedExt> Fgn<T, S> {
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

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::ProcessExt;

  fn lag_covariance(paths: &[Vec<f64>], mean: f64, lag: usize) -> f64 {
    let mut s = 0.0;
    let mut c = 0usize;
    for p in paths {
      for i in 0..(p.len() - lag) {
        s += (p[i] - mean) * (p[i + lag] - mean);
        c += 1;
      }
    }
    s / c as f64
  }

  #[test]
  fn cuda_native_single_path_shape() {
    let fgn = Fgn::<f64>::new(0.7, 1024, Some(1.0));
    let result = fgn
      .sample_cuda_native(1)
      .expect("single path should succeed");
    let path = result.left().expect("m=1 should return Left(Array1)");
    assert_eq!(path.len(), 1024);
  }

  #[test]
  fn cuda_native_batch_shape() {
    let fgn = Fgn::<f64>::new(0.7, 512, Some(1.0));
    let m = 64;
    let result = fgn.sample_cuda_native(m).expect("batch should succeed");
    let batch = result.right().expect("m>1 should return Right(Array2)");
    assert_eq!(batch.shape(), &[m, 512]);
  }

  #[test]
  fn cuda_native_f32_works() {
    let fgn = Fgn::<f32>::new(0.7, 1024, Some(1.0));
    let result = fgn.sample_cuda_native(4).expect("f32 should succeed");
    let batch = result.right().expect("m>1 should return Right(Array2)");
    assert_eq!(batch.shape(), &[4, 1024]);
  }

  #[test]
  fn cuda_native_non_power_of_two_n() {
    let fgn = Fgn::<f64>::new(0.7, 3000, Some(1.0));
    let result = fgn.sample_cuda_native(8).expect("non-pot n should work");
    let batch = result.right().expect("batch");
    assert_eq!(batch.shape(), &[8, 3000]);
  }

  #[test]
  fn cuda_native_eigenvalues_structural() {
    let fgn = Fgn::<f64>::new(0.72, 2048, Some(1.0));
    let eigs = &*fgn.sqrt_eigenvalues;

    assert_eq!(eigs.len(), 2 * fgn.n);
    assert!(eigs.iter().all(|&v| v >= 0.0));

    for i in 1..eigs.len() / 2 {
      let diff = (eigs[i] - eigs[eigs.len() - i]).abs();
      assert!(
        diff < 1e-10,
        "eigs[{i}]={} != eigs[{}]={}",
        eigs[i],
        eigs.len() - i,
        eigs[eigs.len() - i]
      );
    }

    let energy: f64 = eigs.iter().map(|&v| v * v).sum();
    assert!(
      (energy - 1.0).abs() < 1e-6,
      "eigenvalue energy sum should be 1.0, got {energy}"
    );
  }

  #[test]
  fn cuda_native_scale_matches_cpu() {
    for &n in &[1024_usize, 3000, 4096] {
      let fgn = Fgn::<f64>::new(0.7, n, Some(2.0));
      let cpu_scale = fgn.scale;

      let out_size = fgn.n - fgn.offset;
      let scale_steps = out_size.max(1);
      let cuda_scale = (scale_steps as f64).powf(-0.7) * 2.0_f64.powf(0.7);

      assert!(
        (cpu_scale - cuda_scale).abs() < 1e-14,
        "scale mismatch for n={n}: cpu={cpu_scale}, cuda={cuda_scale}"
      );
    }
  }

  #[test]
  fn cuda_native_variance_matches_cpu() {
    let h = 0.72_f64;
    let n = 2048_usize;
    let t = 1.0_f64;
    let m = 1024_usize;
    let fgn = Fgn::<f64>::new(h, n, Some(t));

    let cpu_paths: Vec<Vec<f64>> = (0..m).map(|_| fgn.sample_cpu().to_vec()).collect();
    let cpu_vals: Vec<f64> = cpu_paths.iter().flatten().copied().collect();
    let cpu_mean = cpu_vals.iter().sum::<f64>() / cpu_vals.len() as f64;
    let cpu_var =
      cpu_vals.iter().map(|x| (x - cpu_mean).powi(2)).sum::<f64>() / cpu_vals.len() as f64;

    let cuda_result = fgn
      .sample_cuda_native(m)
      .expect("cuda batch should succeed");
    let cuda_batch = cuda_result.right().expect("batch");
    let cuda_vals: Vec<f64> = cuda_batch.iter().copied().collect();
    let cuda_mean = cuda_vals.iter().sum::<f64>() / cuda_vals.len() as f64;
    let cuda_var = cuda_vals
      .iter()
      .map(|x| (x - cuda_mean).powi(2))
      .sum::<f64>()
      / cuda_vals.len() as f64;

    let ratio = cuda_var / cpu_var;
    assert!(
      (ratio - 1.0).abs() < 0.15,
      "CUDA vs CPU variance ratio = {ratio} (cuda={cuda_var}, cpu={cpu_var})"
    );
  }

  #[test]
  fn cuda_native_covariance_structure_matches_cpu() {
    let h = 0.72_f64;
    let n = 2048_usize;
    let t = 1.0_f64;
    let m = 1024_usize;
    let fgn = Fgn::<f64>::new(h, n, Some(t));

    let cpu_paths: Vec<Vec<f64>> = (0..m).map(|_| fgn.sample_cpu().to_vec()).collect();
    let cpu_vals: Vec<f64> = cpu_paths.iter().flatten().copied().collect();
    let cpu_mean = cpu_vals.iter().sum::<f64>() / cpu_vals.len() as f64;
    let cpu_cov1 = lag_covariance(&cpu_paths, cpu_mean, 1);
    let cpu_cov4 = lag_covariance(&cpu_paths, cpu_mean, 4);

    let cuda_result = fgn
      .sample_cuda_native(m)
      .expect("cuda batch should succeed");
    let cuda_batch = cuda_result.right().expect("batch");
    let cuda_paths: Vec<Vec<f64>> = cuda_batch.rows().into_iter().map(|r| r.to_vec()).collect();
    let cuda_vals: Vec<f64> = cuda_paths.iter().flatten().copied().collect();
    let cuda_mean = cuda_vals.iter().sum::<f64>() / cuda_vals.len() as f64;
    let cuda_cov1 = lag_covariance(&cuda_paths, cuda_mean, 1);
    let cuda_cov4 = lag_covariance(&cuda_paths, cuda_mean, 4);

    let ratio1 = cuda_cov1 / cpu_cov1;
    let ratio4 = cuda_cov4 / cpu_cov4;
    assert!(
      (ratio1 - 1.0).abs() < 0.15,
      "lag-1 cov ratio = {ratio1} (cuda={cuda_cov1}, cpu={cpu_cov1})"
    );
    assert!(
      (ratio4 - 1.0).abs() < 0.15,
      "lag-4 cov ratio = {ratio4} (cuda={cuda_cov4}, cpu={cpu_cov4})"
    );
  }
}
