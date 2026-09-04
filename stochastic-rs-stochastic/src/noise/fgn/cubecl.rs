//! # CubeCL GPU
//!
//! Cross-platform GPU-accelerated Fgn sampling via CubeCL.
//! One text per kernel, launched on whichever runtime the handle names:
//! `CubeclCuda` (cubecl-cuda) or `CubeclWgpu` (cubecl-wgpu — Metal on macOS,
//! Vulkan on Linux, WebGPU on the web).
//!
//! FFT uses shared-memory radix-2 for local stages and radix-4 butterfly
//! for global stages, minimising kernel dispatch count.
//!
use cubecl::prelude::*;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;

use super::Fgn;
use crate::device::DeviceError;
use crate::traits::FloatExt;

type DeviceResult<T> = std::result::Result<T, DeviceError>;

const WG_SIZE: usize = 256;
const BLOCK: usize = WG_SIZE * 2; // 512 elements per shared-memory tile
const LOCAL_STAGES: usize = 9; // log2(512)

/// Shared-memory sub-FFT: loads a contiguous tile of BLOCK elements,
/// performs LOCAL_STAGES radix-2 butterfly stages entirely in shared
/// memory (one sync per stage), then writes back.
#[allow(
  clippy::approx_constant,
  clippy::excessive_precision,
  clippy::identity_op,
  clippy::modulo_one
)]
#[cube(launch)]
fn fft_local<F: Float>(real: &mut Array<F>, imag: &mut Array<F>) {
  let tid = UNIT_POS as usize;
  let base = CUBE_POS * BLOCK;

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
    let a = F::new(-2.0_f32) * F::new(3.141592653589793_f32) * F::cast_from(p) / F::new(2.0_f32);
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
    let a = F::new(-2.0_f32) * F::new(3.141592653589793_f32) * F::cast_from(p) / F::new(4.0_f32);
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
    let a = F::new(-2.0_f32) * F::new(3.141592653589793_f32) * F::cast_from(p) / F::new(8.0_f32);
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
    let a = F::new(-2.0_f32) * F::new(3.141592653589793_f32) * F::cast_from(p) / F::new(16.0_f32);
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
    let a = F::new(-2.0_f32) * F::new(3.141592653589793_f32) * F::cast_from(p) / F::new(32.0_f32);
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
    let a = F::new(-2.0_f32) * F::new(3.141592653589793_f32) * F::cast_from(p) / F::new(64.0_f32);
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
    let a = F::new(-2.0_f32) * F::new(3.141592653589793_f32) * F::cast_from(p) / F::new(128.0_f32);
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
    let a = F::new(-2.0_f32) * F::new(3.141592653589793_f32) * F::cast_from(p) / F::new(256.0_f32);
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
    let a = F::new(-2.0_f32) * F::new(3.141592653589793_f32) * F::cast_from(p) / F::new(512.0_f32);
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

  let a =
    F::new(-2.0_f32) * F::new(3.141592653589793_f32) * F::cast_from(pos) / F::cast_from(stride);
  let (tw_r, tw_i) = (F::cos(a), F::sin(a));
  let (tr, ti) = (
    real[j] * tw_r - imag[j] * tw_i,
    real[j] * tw_i + imag[j] * tw_r,
  );
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

/// GPU normal generation + eigenvalue scale + bit-reversed scatter, fused into
/// one kernel — replaces the host-side RNG and the full host->device upload.
/// Each thread produces one complex sample (two normals via a hash + Box-Muller),
/// scales by its eigenvalue, and writes to the bit-reversed slot.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
#[cube(launch)]
fn gen_scale<F: Float>(
  real: &mut Array<F>,
  imag: &mut Array<F>,
  eigs: &Array<F>,
  rev: &Array<u32>,
  seed: u32,
  first_elem: u32,
  #[comptime] traj_size: usize,
) {
  let tid = ABSOLUTE_POS;
  // Hash on the batch-global element, so chunks continue one launch's stream.
  let g = tid as u32 + first_elem;
  // Two decorrelated uniforms via integer hashing (Murmur3-style finalizer).
  let mut a = (g * 2u32) ^ (seed * 2654435761u32);
  a ^= a >> 16;
  a *= 2246822519u32;
  a ^= a >> 13;
  a *= 3266489917u32;
  a ^= a >> 16;
  let mut b = (g * 2u32 + 1u32) ^ (seed * 668265263u32);
  b ^= b >> 16;
  b *= 2246822519u32;
  b ^= b >> 13;
  b *= 3266489917u32;
  b ^= b >> 16;
  let inv = F::new(2.3283064e-10_f32);
  let u1 = F::cast_from(a) * inv * F::new(0.999998_f32) + F::new(1.0e-6_f32);
  let u2 = F::cast_from(b) * inv;
  let radius = F::sqrt(F::new(-2.0_f32) * F::ln(u1));
  let angle = F::new(6.2831853071_f32) * u2;
  let j = tid % traj_size;
  let e = eigs[j];
  let dst = tid - j + rev[j] as usize;
  real[dst] = radius * F::cos(angle) * e;
  imag[dst] = radius * F::sin(angle) * e;
}

#[cfg(any(feature = "cubecl-cuda", feature = "cubecl-wgpu"))]
mod backend {
  use std::any::TypeId;

  use super::*;
  use crate::euler::cubecl::CubeclRuntime;

  /// Splits a 1D cube count into a 2D grid so no dimension exceeds WebGPU's
  /// 65535 per-dimension limit. For the power-of-two counts this sampler emits
  /// the split is exact (no over-dispatch, so the kernels need no bounds guard).
  fn count_2d(cubes: u32) -> CubeCount {
    if cubes <= 65535 {
      CubeCount::Static(cubes.max(1), 1, 1)
    } else {
      let mut x = cubes;
      let mut y = 1u32;
      while x > 32768 {
        x = x.div_ceil(2);
        y *= 2;
      }
      CubeCount::Static(x, y, 1)
    }
  }

  pub(super) fn sample_cubecl_f32<C: CubeclRuntime, T: FloatExt>(
    sqrt_eigs: &[f32],
    n: usize,
    m: usize,
    offset: usize,
    hurst: f64,
    t: f64,
    first: usize,
    seed_u: u32,
    ordinal: usize,
  ) -> DeviceResult<Array2<T>> {
    let traj_size = 2 * n;
    let out_size = n - offset;
    let scale = (out_size.max(1) as f32).powf(-(hurst as f32)) * (t as f32).powf(hurst as f32);
    let total = m * traj_size;
    let log_n = traj_size.trailing_zeros() as usize;

    let client = C::client(ordinal)?;
    let cl = &client;

    // Bit-reverse table + eigenvalues (small) uploaded; trajectory buffers empty.
    let log_t = traj_size.trailing_zeros() as usize;
    let bits = usize::BITS as usize;
    let rev: Vec<u32> = (0..traj_size)
      .map(|i| (i.reverse_bits() >> (bits - log_t)) as u32)
      .collect();
    let eig_h = cl.create_from_slice(f32::as_bytes(sqrt_eigs));
    let rev_h = cl.create_from_slice(u32::as_bytes(&rev));
    let hr = cl.empty(total * 4);
    let hi = cl.empty(total * 4);

    // GPU: generate normals + eigenvalue scale + bit-reversed scatter.
    unsafe {
      gen_scale::launch::<f32, C::Rt>(
        cl,
        count_2d((total as u32).div_ceil(WG_SIZE as u32)),
        CubeDim::new_1d(WG_SIZE as u32),
        ArrayArg::from_raw_parts::<f32>(&hr, total, 1),
        ArrayArg::from_raw_parts::<f32>(&hi, total, 1),
        ArrayArg::from_raw_parts::<f32>(&eig_h, traj_size, 1),
        ArrayArg::from_raw_parts::<u32>(&rev_h, traj_size, 1),
        ScalarArg::new(seed_u & 0xffff),
        ScalarArg::new((first * traj_size) as u32),
        traj_size,
      )
      .map_err(|e| DeviceError::Launch(format!("gen_scale: {e}")))?;
    }

    // Phase 1: shared-memory local FFT (9 stages per 512-element tile, 1 launch)
    let n_tiles = (total / BLOCK) as u32;
    unsafe {
      fft_local::launch::<f32, C::Rt>(
        cl,
        count_2d(n_tiles),
        CubeDim::new_1d(WG_SIZE as u32),
        ArrayArg::from_raw_parts::<f32>(&hr, total, 1),
        ArrayArg::from_raw_parts::<f32>(&hi, total, 1),
      )
      .map_err(|e| DeviceError::Launch(format!("fft_local: {e}")))?;
    }

    // Phase 2: remaining global stages (LOCAL_STAGES .. log_n)
    let nwg = (total as u32 / 2).div_ceil(WG_SIZE as u32);
    for stage in LOCAL_STAGES..log_n {
      let hs = 1 << stage;
      unsafe {
        fft_butterfly::launch::<f32, C::Rt>(
          cl,
          count_2d(nwg),
          CubeDim::new_1d(WG_SIZE as u32),
          ArrayArg::from_raw_parts::<f32>(&hr, total, 1),
          ArrayArg::from_raw_parts::<f32>(&hi, total, 1),
          traj_size,
          hs,
        )
        .map_err(|e| DeviceError::Launch(format!("fft_butterfly stage {stage}: {e}")))?;
      }
    }

    // Phase 3: extract
    let tout = (m * out_size) as u32;
    let oh = cl.empty(tout as usize * 4);
    let sh = cl.create_from_slice(f32::as_bytes(&[scale]));
    unsafe {
      extract_real::launch::<f32, C::Rt>(
        cl,
        count_2d(tout.div_ceil(WG_SIZE as u32)),
        CubeDim::new_1d(WG_SIZE as u32),
        ArrayArg::from_raw_parts::<f32>(&hr, total, 1),
        ArrayArg::from_raw_parts::<f32>(&oh, m * out_size, 1),
        ArrayArg::from_raw_parts::<f32>(&sh, 1, 1),
        out_size,
        traj_size,
      )
      .map_err(|e| DeviceError::Launch(format!("extract_real: {e}")))?;
    }

    let bytes = cl.read_one(oh.clone());
    let out = f32::from_bytes(&bytes);
    let fgn = arr2::<T>(out, m, out_size);
    Ok(fgn)
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

impl<T: FloatExt, S: SeedExt, B> Fgn<T, S, B> {
  /// `m` paths on one CubeCL runtime, in chunks that fit the batch
  /// budget: one seed for the whole batch and a running element offset in the
  /// kernel's hash, so the result is the same whatever the budget.
  #[cfg(any(feature = "cubecl-cuda", feature = "cubecl-wgpu"))]
  fn sample_cubecl_on<C: crate::euler::cubecl::CubeclRuntime, S2: SeedExt>(
    &self,
    m: usize,
    seed_src: &S2,
    ordinal: usize,
    batch_budget: usize,
  ) -> DeviceResult<Array2<T>> {
    let n = self.n;
    let offset = self.offset;
    let out_size = n - offset;
    let hurst = self.hurst.to_f64().unwrap();
    let t = self.t.unwrap_or(T::one()).to_f64().unwrap();
    let eigs: Vec<f32> = self
      .sqrt_eigenvalues
      .iter()
      .map(|x| x.to_f32().unwrap())
      .collect();
    let seed_u: u32 = rand::Rng::random(&mut seed_src.rng());
    let rows = crate::device::chunk_rows(batch_budget, 4 * n + out_size, 4);
    let mut out = Array2::<T>::zeros((m, out_size));
    let mut first = 0;
    while first < m {
      let len = rows.min(m - first);
      let chunk = backend::sample_cubecl_f32::<C, T>(
        &eigs, n, len, offset, hurst, t, first, seed_u, ordinal,
      )?;
      out
        .slice_mut(ndarray::s![first..first + len, ..])
        .assign(&chunk);
      first += len;
    }
    Ok(out)
  }

  /// `m` paths on the runtime the handle names.
  #[cfg(any(feature = "cubecl-cuda", feature = "cubecl-wgpu"))]
  pub(crate) fn sample_cubecl_impl<R: crate::euler::cubecl::CubeclRuntime, S2: SeedExt>(
    &self,
    m: usize,
    seed_src: &S2,
    device: &crate::device::Cubecl<R>,
  ) -> DeviceResult<Array2<T>> {
    self.sample_cubecl_on::<R, S2>(m, seed_src, device.ordinal, device.batch_budget)
  }
}
