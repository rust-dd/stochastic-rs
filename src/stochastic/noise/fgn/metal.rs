//! # Metal GPU
//!
//! macOS Metal compute backend for FGN sampling.
//! MSL shaders compiled at runtime, unified memory (zero-copy on Apple Silicon),
//! all FFT stages encoded into a single command buffer.
//!
use std::any::TypeId;

use anyhow::Result;
use either::Either;
use metal::*;
use ndarray::{Array1, Array2};
use parking_lot::Mutex;

use super::FGN;
use crate::simd_rng::SeedExt;
use crate::traits::FloatExt;

const MSL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fft_butterfly(
    device float* real [[buffer(0)]],
    device float* imag [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& half_stride [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint butterflies_per_batch = n / 2;
    uint batch = tid / butterflies_per_batch;
    uint local_tid = tid % butterflies_per_batch;
    uint stride = half_stride * 2;
    uint group = local_tid / half_stride;
    uint pos = local_tid % half_stride;
    uint base = batch * n;
    uint i = base + group * stride + pos;
    uint j = i + half_stride;

    float angle = -2.0f * 3.14159265358979323846f * float(pos) / float(stride);
    float tw_r = cos(angle);
    float tw_i = sin(angle);

    float tr = real[j] * tw_r - imag[j] * tw_i;
    float ti = real[j] * tw_i + imag[j] * tw_r;
    float ar = real[i];
    float ai = imag[i];

    real[i] = ar + tr;
    imag[i] = ai + ti;
    real[j] = ar - tr;
    imag[j] = ai - ti;
}

kernel void extract_real(
    device const float* src_real [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& out_size [[buffer(2)]],
    constant uint& traj_size [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint traj_id = tid / out_size;
    uint idx = tid % out_size;
    output[tid] = src_real[traj_id * traj_size + idx + 1] * scale;
}
"#;

struct MetalCtx {
  device: Device,
  queue: CommandQueue,
  butterfly_pso: ComputePipelineState,
  extract_pso: ComputePipelineState,
}

unsafe impl Send for MetalCtx {}

static CTX: Mutex<Option<MetalCtx>> = Mutex::new(None);

fn ensure_ctx() -> Result<()> {
  let mut g = CTX.lock();
  if g.is_some() {
    return Ok(());
  }
  let device = Device::system_default().ok_or_else(|| anyhow::anyhow!("no Metal device"))?;
  let queue = device.new_command_queue();
  let lib = device
    .new_library_with_source(MSL_SOURCE, &CompileOptions::new())
    .map_err(|e| anyhow::anyhow!("MSL compile: {e}"))?;
  let bf_fn = lib
    .get_function("fft_butterfly", None)
    .map_err(|e| anyhow::anyhow!("get fft_butterfly: {e}"))?;
  let ex_fn = lib
    .get_function("extract_real", None)
    .map_err(|e| anyhow::anyhow!("get extract_real: {e}"))?;
  let butterfly_pso = device
    .new_compute_pipeline_state_with_function(&bf_fn)
    .map_err(|e| anyhow::anyhow!("butterfly PSO: {e}"))?;
  let extract_pso = device
    .new_compute_pipeline_state_with_function(&ex_fn)
    .map_err(|e| anyhow::anyhow!("extract PSO: {e}"))?;
  *g = Some(MetalCtx { device, queue, butterfly_pso, extract_pso });
  Ok(())
}

fn sample_f32<T: FloatExt>(
  sqrt_eigs: &[f32],
  n: usize,
  m: usize,
  offset: usize,
  hurst: f64,
  t: f64,
) -> Result<Either<Array1<T>, Array2<T>>> {
  let traj_size = 2 * n;
  let out_size = n - offset;
  let scale = (out_size.max(1) as f32).powf(-(hurst as f32)) * (t as f32).powf(hurst as f32);
  let total = m * traj_size;
  let log_n = traj_size.trailing_zeros() as usize;

  // CPU: normals + eigenvalue scaling + bit-reversal
  let mut real_host = vec![0.0f32; total];
  let mut imag_host = vec![0.0f32; total];
  {
    let normal = crate::distributions::normal::SimdNormal::<f32>::new(0.0, 1.0);
    normal.fill_slice_fast(&mut real_host);
    normal.fill_slice_fast(&mut imag_host);
  }
  for i in 0..total {
    let e = sqrt_eigs[i % traj_size];
    real_host[i] *= e;
    imag_host[i] *= e;
  }
  bit_reverse_permute(&mut real_host, &mut imag_host, traj_size, m);

  ensure_ctx()?;
  let g = CTX.lock();
  let ctx = g.as_ref().unwrap();

  // Shared-mode buffers: zero-copy on Apple Silicon unified memory
  let bytes = (total * 4) as u64;
  let real_buf = ctx.device.new_buffer_with_data(
    real_host.as_ptr() as *const _,
    bytes,
    MTLResourceOptions::StorageModeShared,
  );
  let imag_buf = ctx.device.new_buffer_with_data(
    imag_host.as_ptr() as *const _,
    bytes,
    MTLResourceOptions::StorageModeShared,
  );

  let out_bytes = (m * out_size * 4) as u64;
  let out_buf = ctx
    .device
    .new_buffer(out_bytes, MTLResourceOptions::StorageModeShared);

  // Encode ALL FFT stages + extract into a SINGLE command buffer
  let cmd = ctx.queue.new_command_buffer();

  let n_u32 = traj_size as u32;
  let tg_size = MTLSize::new(256, 1, 1);
  let butterflies = (total / 2) as u64;
  let grid_fft = MTLSize::new(butterflies, 1, 1);

  for stage in 0..log_n {
    let hs = (1u32 << stage) as u32;
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.butterfly_pso);
    enc.set_buffer(0, Some(&real_buf), 0);
    enc.set_buffer(1, Some(&imag_buf), 0);
    enc.set_bytes(2, 4, &n_u32 as *const u32 as *const _);
    enc.set_bytes(3, 4, &hs as *const u32 as *const _);
    enc.dispatch_threads(grid_fft, tg_size);
    enc.end_encoding();
  }

  // Extract
  {
    let os = out_size as u32;
    let ts = traj_size as u32;
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.extract_pso);
    enc.set_buffer(0, Some(&real_buf), 0);
    enc.set_buffer(1, Some(&out_buf), 0);
    enc.set_bytes(2, 4, &os as *const u32 as *const _);
    enc.set_bytes(3, 4, &ts as *const u32 as *const _);
    enc.set_bytes(4, 4, &scale as *const f32 as *const _);
    enc.dispatch_threads(MTLSize::new((m * out_size) as u64, 1, 1), tg_size);
    enc.end_encoding();
  }

  cmd.commit();
  cmd.wait_until_completed();

  // Read back from shared buffer (zero-copy, just pointer cast)
  let out_ptr = out_buf.contents() as *const f32;
  let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, m * out_size) };

  let fgn = arr2_f32::<T>(out_slice, m, out_size);
  if m == 1 {
    return Ok(Either::Left(fgn.row(0).to_owned()));
  }
  Ok(Either::Right(fgn))
}

fn bit_reverse_permute(real: &mut [f32], imag: &mut [f32], n: usize, m: usize) {
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

fn arr2_f32<T: FloatExt>(data: &[f32], m: usize, cols: usize) -> Array2<T> {
  if TypeId::of::<T>() == TypeId::of::<f32>() {
    let out = Array2::<f32>::from_shape_vec((m, cols), data.to_vec()).expect("shape");
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

impl<T: FloatExt, S: SeedExt> FGN<T, S> {
  pub(crate) fn sample_metal_impl(&self, m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    let n = self.n;
    let offset = self.offset;
    let hurst = self.hurst.to_f64().unwrap();
    let t = self.t.unwrap_or(T::one()).to_f64().unwrap();
    let eigs: Vec<f32> = self.sqrt_eigenvalues.iter().map(|x| x.to_f32().unwrap()).collect();
    sample_f32::<T>(&eigs, n, m, offset, hurst, t)
  }
}
