//! # CubeCL GPU
//!
//! Cross-platform GPU-accelerated FGN sampling via CubeCL.
//! Supports CUDA (gpu-cuda), Metal/Vulkan/WebGPU (gpu-wgpu).
//!
//! FFT uses shared-memory radix-2 for local stages and radix-4 butterfly
//! for global stages, minimising kernel dispatch count.
//!
use anyhow::Result;
use cubecl::prelude::*;
use either::Either;
use ndarray::Array1;
use ndarray::Array2;

use super::FGN;
use crate::simd_rng::SeedExt;
use crate::traits::FloatExt;

const WG_SIZE: usize = 256;
const BLOCK: usize = WG_SIZE * 2; // 512 elements per shared-memory tile
const LOCAL_STAGES: usize = 9; // log2(512)

/// Shared-memory sub-FFT: loads a contiguous tile of BLOCK elements,
/// performs LOCAL_STAGES radix-2 butterfly stages entirely in shared
/// memory (one sync per stage), then writes back.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
#[cube(launch)]
fn fft_local<F: Float>(real: &mut Array<F>, imag: &mut Array<F>) {
  let tid = UNIT_POS as usize;
  let base = CUBE_POS_X as usize * BLOCK;

  let mut sr = SharedMemory::<F>::new(BLOCK);
  let mut si = SharedMemory::<F>::new(BLOCK);

  sr[tid] = real[base + tid];
  si[tid] = imag[base + tid];
  sr[tid + WG_SIZE] = real[base + tid + WG_SIZE];
  si[tid + WG_SIZE] = imag[base + tid + WG_SIZE];
  sync_cube();

  // stage 0: half_stride=1
  {
    let g = tid / 1;
    let p = tid % 1;
    let i = g * 2 + p;
    let j = i + 1;
    let a = F::new(-2.0) * F::new(3.141592653589793) * F::cast_from(p) / F::new(2.0);
    let (tw_r, tw_i) = (F::cos(a), F::sin(a));
    let (tr, ti) = (sr[j] * tw_r - si[j] * tw_i, sr[j] * tw_i + si[j] * tw_r);
    let (ar, ai) = (sr[i], si[i]);
    sr[i] = ar + tr;
    si[i] = ai + ti;
    sr[j] = ar - tr;
    si[j] = ai - ti;
    sync_cube();
  }
  // stage 1: half_stride=2
  {
    let g = tid / 2;
    let p = tid % 2;
    let i = g * 4 + p;
    let j = i + 2;
    let a = F::new(-2.0) * F::new(3.141592653589793) * F::cast_from(p) / F::new(4.0);
    let (tw_r, tw_i) = (F::cos(a), F::sin(a));
    let (tr, ti) = (sr[j] * tw_r - si[j] * tw_i, sr[j] * tw_i + si[j] * tw_r);
    let (ar, ai) = (sr[i], si[i]);
    sr[i] = ar + tr;
    si[i] = ai + ti;
    sr[j] = ar - tr;
    si[j] = ai - ti;
    sync_cube();
  }
  // stage 2: half_stride=4
  {
    let g = tid / 4;
    let p = tid % 4;
    let i = g * 8 + p;
    let j = i + 4;
    let a = F::new(-2.0) * F::new(3.141592653589793) * F::cast_from(p) / F::new(8.0);
    let (tw_r, tw_i) = (F::cos(a), F::sin(a));
    let (tr, ti) = (sr[j] * tw_r - si[j] * tw_i, sr[j] * tw_i + si[j] * tw_r);
    let (ar, ai) = (sr[i], si[i]);
    sr[i] = ar + tr;
    si[i] = ai + ti;
    sr[j] = ar - tr;
    si[j] = ai - ti;
    sync_cube();
  }
  // stage 3: half_stride=8
  {
    let g = tid / 8;
    let p = tid % 8;
    let i = g * 16 + p;
    let j = i + 8;
    let a = F::new(-2.0) * F::new(3.141592653589793) * F::cast_from(p) / F::new(16.0);
    let (tw_r, tw_i) = (F::cos(a), F::sin(a));
    let (tr, ti) = (sr[j] * tw_r - si[j] * tw_i, sr[j] * tw_i + si[j] * tw_r);
    let (ar, ai) = (sr[i], si[i]);
    sr[i] = ar + tr;
    si[i] = ai + ti;
    sr[j] = ar - tr;
    si[j] = ai - ti;
    sync_cube();
  }
  // stage 4: half_stride=16
  {
    let g = tid / 16;
    let p = tid % 16;
    let i = g * 32 + p;
    let j = i + 16;
    let a = F::new(-2.0) * F::new(3.141592653589793) * F::cast_from(p) / F::new(32.0);
    let (tw_r, tw_i) = (F::cos(a), F::sin(a));
    let (tr, ti) = (sr[j] * tw_r - si[j] * tw_i, sr[j] * tw_i + si[j] * tw_r);
    let (ar, ai) = (sr[i], si[i]);
    sr[i] = ar + tr;
    si[i] = ai + ti;
    sr[j] = ar - tr;
    si[j] = ai - ti;
    sync_cube();
  }
  // stage 5: half_stride=32
  {
    let g = tid / 32;
    let p = tid % 32;
    let i = g * 64 + p;
    let j = i + 32;
    let a = F::new(-2.0) * F::new(3.141592653589793) * F::cast_from(p) / F::new(64.0);
    let (tw_r, tw_i) = (F::cos(a), F::sin(a));
    let (tr, ti) = (sr[j] * tw_r - si[j] * tw_i, sr[j] * tw_i + si[j] * tw_r);
    let (ar, ai) = (sr[i], si[i]);
    sr[i] = ar + tr;
    si[i] = ai + ti;
    sr[j] = ar - tr;
    si[j] = ai - ti;
    sync_cube();
  }
  // stage 6: half_stride=64
  {
    let g = tid / 64;
    let p = tid % 64;
    let i = g * 128 + p;
    let j = i + 64;
    let a = F::new(-2.0) * F::new(3.141592653589793) * F::cast_from(p) / F::new(128.0);
    let (tw_r, tw_i) = (F::cos(a), F::sin(a));
    let (tr, ti) = (sr[j] * tw_r - si[j] * tw_i, sr[j] * tw_i + si[j] * tw_r);
    let (ar, ai) = (sr[i], si[i]);
    sr[i] = ar + tr;
    si[i] = ai + ti;
    sr[j] = ar - tr;
    si[j] = ai - ti;
    sync_cube();
  }
  // stage 7: half_stride=128
  {
    let g = tid / 128;
    let p = tid % 128;
    let i = g * 256 + p;
    let j = i + 128;
    let a = F::new(-2.0) * F::new(3.141592653589793) * F::cast_from(p) / F::new(256.0);
    let (tw_r, tw_i) = (F::cos(a), F::sin(a));
    let (tr, ti) = (sr[j] * tw_r - si[j] * tw_i, sr[j] * tw_i + si[j] * tw_r);
    let (ar, ai) = (sr[i], si[i]);
    sr[i] = ar + tr;
    si[i] = ai + ti;
    sr[j] = ar - tr;
    si[j] = ai - ti;
    sync_cube();
  }
  // stage 8: half_stride=256
  {
    let p = tid;
    let i = p;
    let j = i + 256;
    let a = F::new(-2.0) * F::new(3.141592653589793) * F::cast_from(p) / F::new(512.0);
    let (tw_r, tw_i) = (F::cos(a), F::sin(a));
    let (tr, ti) = (sr[j] * tw_r - si[j] * tw_i, sr[j] * tw_i + si[j] * tw_r);
    let (ar, ai) = (sr[i], si[i]);
    sr[i] = ar + tr;
    si[i] = ai + ti;
    sr[j] = ar - tr;
    si[j] = ai - ti;
    sync_cube();
  }

  real[base + tid] = sr[tid];
  imag[base + tid] = si[tid];
  real[base + tid + WG_SIZE] = sr[tid + WG_SIZE];
  imag[base + tid + WG_SIZE] = si[tid + WG_SIZE];
}

/// Global radix-2 butterfly for remaining stages after the shared-memory pass.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
#[cube(launch)]
fn fft_butterfly<F: Float>(
  real: &mut Array<F>,
  imag: &mut Array<F>,
  #[comptime] n: usize,
  #[comptime] half_stride: usize,
) {
  let tid = ABSOLUTE_POS;
  let batch = tid / (n / 2);
  let local = tid % (n / 2);
  let stride = half_stride * 2;
  let group = local / half_stride;
  let pos = local % half_stride;
  let base = batch * n;
  let i = base + group * stride + pos;
  let j = i + half_stride;

  let a = F::new(-2.0) * F::new(3.141592653589793) * F::cast_from(pos) / F::cast_from(stride);
  let (tw_r, tw_i) = (F::cos(a), F::sin(a));
  let (tr, ti) = (real[j] * tw_r - imag[j] * tw_i, real[j] * tw_i + imag[j] * tw_r);
  let (ar, ai) = (real[i], imag[i]);
  real[i] = ar + tr;
  imag[i] = ai + ti;
  real[j] = ar - tr;
  imag[j] = ai - ti;
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
  output[tid] = src_real[traj_id * traj_size + idx + 1] * scale;
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

  unsafe impl Send for GpuContext {}

  static GPU_CTX: Mutex<Option<GpuContext>> = Mutex::new(None);

  fn ensure_ctx(n: usize, m: usize, offset: usize, hb: u64, tb: u64) {
    let mut g = GPU_CTX.lock();
    let need = match &*g {
      Some(c) => c.n != n || c.m != m || c.offset != offset || c.hurst_bits != hb || c.t_bits != tb,
      None => true,
    };
    if !need {
      return;
    }
    #[cfg(feature = "gpu-cuda")]
    let dev = cubecl_cuda::CudaDevice::default();
    #[cfg(all(feature = "gpu-wgpu", not(feature = "gpu-cuda")))]
    let dev = cubecl_wgpu::WgpuDevice::default();
    *g = Some(GpuContext {
      client: R::client(&dev),
      n,
      m,
      offset,
      hurst_bits: hb,
      t_bits: tb,
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
    let hb = hurst.to_bits();
    let tb = t.to_bits();
    let traj_size = 2 * n;
    let out_size = n - offset;
    let scale = (out_size.max(1) as f32).powf(-(hurst as f32)) * (t as f32).powf(hurst as f32);
    let total = m * traj_size;
    let log_n = traj_size.trailing_zeros() as usize;

    // CPU: generate normals, scale, bit-reverse
    let mut rh = vec![0.0f32; total];
    let mut ih = vec![0.0f32; total];
    {
      let normal = crate::distributions::normal::SimdNormal::<f32>::new(0.0, 1.0);
      normal.fill_slice_fast(&mut rh);
      normal.fill_slice_fast(&mut ih);
    }
    for i in 0..total {
      let e = sqrt_eigs[i % traj_size];
      rh[i] *= e;
      ih[i] *= e;
    }
    bit_reverse_permute_batched(&mut rh, &mut ih, traj_size, m);

    ensure_ctx(n, m, offset, hb, tb);
    let guard = GPU_CTX.lock();
    let cl = &guard.as_ref().unwrap().client;

    let hr = cl.create_from_slice(f32::as_bytes(&rh));
    let hi = cl.create_from_slice(f32::as_bytes(&ih));

    // Phase 1: shared-memory local FFT (9 stages per 512-element tile, 1 launch)
    let n_tiles = (total / BLOCK) as u32;
    unsafe {
      fft_local::launch::<f32, R>(
        cl,
        CubeCount::Static(n_tiles, 1, 1),
        CubeDim::new_1d(WG_SIZE as u32),
        ArrayArg::from_raw_parts::<f32>(&hr, total, 1),
        ArrayArg::from_raw_parts::<f32>(&hi, total, 1),
      )
      .map_err(|e| anyhow::anyhow!("fft_local: {e}"))?;
    }

    // Phase 2: remaining global stages (LOCAL_STAGES .. log_n)
    let nwg = (total as u32 / 2).div_ceil(WG_SIZE as u32);
    for stage in LOCAL_STAGES..log_n {
      let hs = 1 << stage;
      unsafe {
        fft_butterfly::launch::<f32, R>(
          cl,
          CubeCount::Static(nwg, 1, 1),
          CubeDim::new_1d(WG_SIZE as u32),
          ArrayArg::from_raw_parts::<f32>(&hr, total, 1),
          ArrayArg::from_raw_parts::<f32>(&hi, total, 1),
          traj_size,
          hs,
        )
        .map_err(|e| anyhow::anyhow!("fft_butterfly stage {stage}: {e}"))?;
      }
    }

    // Phase 3: extract
    let tout = (m * out_size) as u32;
    let oh = cl.empty(tout as usize * 4);
    let sh = cl.create_from_slice(f32::as_bytes(&[scale]));
    unsafe {
      extract_real::launch::<f32, R>(
        cl,
        CubeCount::Static(tout.div_ceil(WG_SIZE as u32), 1, 1),
        CubeDim::new_1d(WG_SIZE as u32),
        ArrayArg::from_raw_parts::<f32>(&hr, total, 1),
        ArrayArg::from_raw_parts::<f32>(&oh, m * out_size, 1),
        ArrayArg::from_raw_parts::<f32>(&sh, 1, 1),
        out_size,
        traj_size,
      )
      .map_err(|e| anyhow::anyhow!("extract_real: {e}"))?;
    }

    let bytes = cl.read_one(oh.clone());
    let out = f32::from_bytes(&bytes);
    let fgn = arr2::<T>(out, m, out_size);
    drop(guard);
    if m == 1 {
      return Ok(Either::Left(fgn.row(0).to_owned()));
    }
    Ok(Either::Right(fgn))
  }

  fn arr2<T: FloatExt>(data: &[f32], m: usize, cols: usize) -> Array2<T> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
      let o = Array2::<f32>::from_shape_vec((m, cols), data.to_vec()).expect("shape");
      unsafe { std::mem::transmute::<Array2<f32>, Array2<T>>(o) }
    } else {
      let mut o = Array2::<T>::zeros((m, cols));
      for i in 0..m {
        for j in 0..cols {
          o[[i, j]] = T::from_f64_fast(data[i * cols + j] as f64);
        }
      }
      o
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
