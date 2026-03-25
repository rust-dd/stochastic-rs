//! # CubeCL GPU
//!
//! Cross-platform GPU-accelerated FGN sampling via CubeCL.
//! Supports CUDA (gpu-cuda), Metal/Vulkan/WebGPU (gpu-wgpu).
//!
//! Pipeline: CPU SIMD RNG + eigenvalue scaling + bit-reversal -> upload ->
//! radix-2 Cooley-Tukey FFT (CubeCL kernels) -> extract -> download.
//! Context (device + pre-uploaded tables) is cached across calls.
//!
use anyhow::Result;
use cubecl::prelude::*;
use either::Either;
use ndarray::Array1;
use ndarray::Array2;

use super::FGN;
use crate::simd_rng::SeedExt;
use crate::traits::FloatExt;

/// One stage of a radix-2 DIT Cooley-Tukey butterfly.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
#[cube(launch)]
fn fft_butterfly<F: Float>(
  real: &mut Array<F>,
  imag: &mut Array<F>,
  #[comptime] n: usize,
  #[comptime] half_stride: usize,
) {
  let tid = ABSOLUTE_POS;
  let butterflies_per_batch = n / 2;
  let batch = tid / butterflies_per_batch;
  let local = tid % butterflies_per_batch;

  let stride = half_stride * 2;
  let group = local / half_stride;
  let pos = local % half_stride;

  let base = batch * n;
  let i = base + group * stride + pos;
  let j = i + half_stride;

  let angle = F::new(-2.0) * F::new(3.141592653589793) * F::cast_from(pos) / F::cast_from(stride);
  let tw_re = F::cos(angle);
  let tw_im = F::sin(angle);

  let t_re = real[j] * tw_re - imag[j] * tw_im;
  let t_im = real[j] * tw_im + imag[j] * tw_re;

  let a_re = real[i];
  let a_im = imag[i];
  real[i] = a_re + t_re;
  imag[i] = a_im + t_im;
  real[j] = a_re - t_re;
  imag[j] = a_im - t_im;
}

/// Extract real parts from FFT output with offset and scaling.
#[cube(launch)]
fn extract_real<F: Float>(
  src_real: &Array<F>,
  output: &mut Array<F>,
  scale_arr: &Array<F>,
  #[comptime] out_size: usize,
  #[comptime] traj_size: usize,
) {
  let tid = ABSOLUTE_POS;
  let scale = scale_arr[0];
  let traj_id = tid / out_size;
  let idx = tid % out_size;
  let src_idx = traj_id * traj_size + idx + 1;
  output[tid] = src_real[src_idx] * scale;
}

fn bit_reverse_permute_batched(real: &mut [f32], imag: &mut [f32], n: usize, m: usize) {
  let log_n = n.trailing_zeros() as usize;
  let bits = usize::BITS as usize;
  for b in 0..m {
    let base = b * n;
    for i in 0..n {
      let j = i.reverse_bits() >> (bits - log_n);
      if i < j {
        real.swap(base + i, base + j);
        imag.swap(base + i, base + j);
      }
    }
  }
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu"))]
mod backend {
  use std::any::TypeId;

  use cubecl::client::ComputeClient;
  use parking_lot::Mutex;

  use super::*;

  #[cfg(feature = "gpu-cuda")]
  pub(super) type R = cubecl_cuda::CudaRuntime;

  #[cfg(all(feature = "gpu-wgpu", not(feature = "gpu-cuda")))]
  pub(super) type R = cubecl_wgpu::WgpuRuntime;

  struct GpuContext {
    client: ComputeClient<R>,
    n: usize,
    m: usize,
    offset: usize,
    hurst_bits: u64,
    t_bits: u64,
  }

  // SAFETY: CubeCL clients are internally synchronized.
  unsafe impl Send for GpuContext {}

  static GPU_CTX: Mutex<Option<GpuContext>> = Mutex::new(None);

  fn ensure_ctx(n: usize, m: usize, offset: usize, hurst_bits: u64, t_bits: u64) {
    let mut guard = GPU_CTX.lock();
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
    if !need_init {
      return;
    }

    #[cfg(feature = "gpu-cuda")]
    let device = cubecl_cuda::CudaDevice::default();
    #[cfg(all(feature = "gpu-wgpu", not(feature = "gpu-cuda")))]
    let device = cubecl_wgpu::WgpuDevice::default();

    *guard = Some(GpuContext {
      client: R::client(&device),
      n,
      m,
      offset,
      hurst_bits,
      t_bits,
    });
  }

  pub(super) fn sample_gpu_f32<T: FloatExt>(
    sqrt_eigs: &[f32],
    n: usize,
    m: usize,
    offset: usize,
    hurst: f64,
    t: f64,
  ) -> Result<Either<Array1<T>, Array2<T>>> {
    let hurst_bits = hurst.to_bits();
    let t_bits = t.to_bits();
    let traj_size = 2 * n;
    let out_size = n - offset;
    let scale_steps = out_size.max(1);
    let scale = (scale_steps as f32).powf(-(hurst as f32)) * (t as f32).powf(hurst as f32);
    let total = m * traj_size;
    let log_n = traj_size.trailing_zeros() as usize;
    let threads_per_wg = 256u32;

    // CPU: generate normals, scale by eigenvalues, bit-reverse
    let mut real_host = vec![0.0f32; total];
    let mut imag_host = vec![0.0f32; total];
    {
      let normal = crate::distributions::normal::SimdNormal::<f32>::new(0.0, 1.0);
      normal.fill_slice_fast(&mut real_host);
      normal.fill_slice_fast(&mut imag_host);
    }
    for i in 0..total {
      let eig = sqrt_eigs[i % traj_size];
      real_host[i] *= eig;
      imag_host[i] *= eig;
    }
    bit_reverse_permute_batched(&mut real_host, &mut imag_host, traj_size, m);

    ensure_ctx(n, m, offset, hurst_bits, t_bits);
    let guard = GPU_CTX.lock();
    let ctx = guard.as_ref().unwrap();
    let client = &ctx.client;

    // Upload
    let real_handle = client.create_from_slice(f32::as_bytes(&real_host));
    let imag_handle = client.create_from_slice(f32::as_bytes(&imag_host));

    // FFT butterfly stages
    let num_wg_fft = (total as u32 / 2).div_ceil(threads_per_wg);
    for stage in 0..log_n {
      let half_stride = 1 << stage;
      unsafe {
        fft_butterfly::launch::<f32, R>(
          client,
          CubeCount::Static(num_wg_fft, 1, 1),
          CubeDim::new_1d(threads_per_wg),
          ArrayArg::from_raw_parts::<f32>(&real_handle, total, 1),
          ArrayArg::from_raw_parts::<f32>(&imag_handle, total, 1),
          traj_size,
          half_stride,
        )
        .map_err(|e| anyhow::anyhow!("FFT stage {stage}: {e}"))?;
      }
    }

    // Extract real parts + scale
    let total_out = (m * out_size) as u32;
    let out_handle = client.empty(total_out as usize * std::mem::size_of::<f32>());
    let scale_handle = client.create_from_slice(f32::as_bytes(&[scale]));
    let num_wg_out = total_out.div_ceil(threads_per_wg);
    unsafe {
      extract_real::launch::<f32, R>(
        client,
        CubeCount::Static(num_wg_out, 1, 1),
        CubeDim::new_1d(threads_per_wg),
        ArrayArg::from_raw_parts::<f32>(&real_handle, total, 1),
        ArrayArg::from_raw_parts::<f32>(&out_handle, m * out_size, 1),
        ArrayArg::from_raw_parts::<f32>(&scale_handle, 1, 1),
        out_size,
        traj_size,
      )
      .map_err(|e| anyhow::anyhow!("extract_real: {e}"))?;
    }

    // Download
    let out_bytes = client.read_one(out_handle.clone());
    let out_slice = f32::from_bytes(&out_bytes);
    let fgn = array2_from_slice_f32::<T>(out_slice, m, out_size);
    drop(guard);

    if m == 1 {
      return Ok(Either::Left(fgn.row(0).to_owned()));
    }
    Ok(Either::Right(fgn))
  }

  fn array2_from_slice_f32<T: FloatExt>(data: &[f32], m: usize, cols: usize) -> Array2<T> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
      let out =
        Array2::<f32>::from_shape_vec((m, cols), data.to_vec()).expect("shape must be valid");
      unsafe { std::mem::transmute::<Array2<f32>, Array2<T>>(out) }
    } else {
      let mut out = Array2::<T>::zeros((m, cols));
      for i in 0..m {
        for j in 0..cols {
          out[[i, j]] = T::from_f64_fast(data[i * cols + j] as f64);
        }
      }
      out
    }
  }
}

impl<T: FloatExt, S: SeedExt> FGN<T, S> {
  pub(crate) fn sample_gpu_impl(&self, m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    #[cfg(not(any(feature = "gpu-cuda", feature = "gpu-wgpu")))]
    {
      let _ = m;
      anyhow::bail!("No GPU backend selected. Enable `gpu-cuda` or `gpu-wgpu` feature.")
    }

    #[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu"))]
    {
      let n = self.n;
      let offset = self.offset;
      let hurst = self.hurst.to_f64().unwrap();
      let t = self.t.unwrap_or(T::one()).to_f64().unwrap();
      let eigs: Vec<f32> = self
        .sqrt_eigenvalues
        .iter()
        .map(|x| x.to_f32().unwrap())
        .collect();
      backend::sample_gpu_f32::<T>(&eigs, n, m, offset, hurst, t)
    }
  }
}
