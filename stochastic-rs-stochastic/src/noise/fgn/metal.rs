//! # Metal GPU
//!
//! macOS Metal compute backend for Fgn sampling.
//! Full GPU pipeline: Philox RNG + Box-Muller -> eigenvalue scale ->
//! bit-reversal -> FFT butterfly stages -> extract.
//! All encoded into a single command buffer, unified memory zero-copy.
//!
use std::any::TypeId;

use anyhow::Result;
use either::Either;
use metal::*;
use ndarray::Array1;
use ndarray::Array2;
use parking_lot::Mutex;

use super::Fgn;
use stochastic_rs_core::simd_rng::SeedExt;
use crate::traits::FloatExt;

const MSL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// PCG hash: excellent statistical quality, single-instruction path
inline uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

inline float u01(uint x) {
    return (float(x >> 8) + 0.5f) / 16777216.0f;
}

// Generate normals, scale by eigenvalues, write to bit-reversed position.
// Each thread produces one (re, im) pair using 4 independent PCG hashes
// fed into two Box-Muller transforms.
kernel void generate_scale_permute(
    device float* dst_real [[buffer(0)]],
    device float* dst_imag [[buffer(1)]],
    device const float* sqrt_eigs [[buffer(2)]],
    device const uint* bit_rev [[buffer(3)]],
    constant uint& traj_size [[buffer(4)]],
    constant uint& seed [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    uint batch = tid / traj_size;
    uint local = tid % traj_size;

    uint base = tid * 4u + seed;
    float u1 = u01(pcg(base));
    float u2 = u01(pcg(base + 1u));
    float u3 = u01(pcg(base + 2u));
    float u4 = u01(pcg(base + 3u));

    float r_a = sqrt(-2.0f * log(u1 + 1e-10f));
    float r_b = sqrt(-2.0f * log(u3 + 1e-10f));
    float n_re = r_a * cos(6.28318530718f * u2);
    float n_im = r_b * cos(6.28318530718f * u4);

    float eig = sqrt_eigs[local];
    uint rev = bit_rev[local];
    uint dst = batch * traj_size + rev;
    dst_real[dst] = n_re * eig;
    dst_imag[dst] = n_im * eig;
}

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
  gen_pso: ComputePipelineState,
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

  let mk = |name: &str| -> Result<ComputePipelineState> {
    let f = lib
      .get_function(name, None)
      .map_err(|e| anyhow::anyhow!("get {name}: {e}"))?;
    device
      .new_compute_pipeline_state_with_function(&f)
      .map_err(|e| anyhow::anyhow!("{name} PSO: {e}"))
  };

  let gen_pso = mk("generate_scale_permute")?;
  let butterfly_pso = mk("fft_butterfly")?;
  let extract_pso = mk("extract_real")?;

  *g = Some(MetalCtx {
    device,
    queue,
    gen_pso,
    butterfly_pso,
    extract_pso,
  });
  Ok(())
}

fn build_bit_reverse_table(n: usize) -> Vec<u32> {
  let log_n = n.trailing_zeros() as usize;
  let bits = usize::BITS as usize;
  (0..n)
    .map(|i| (i.reverse_bits() >> (bits - log_n)) as u32)
    .collect()
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

  ensure_ctx()?;
  let g = CTX.lock();
  let ctx = g.as_ref().unwrap();
  let dev = &ctx.device;
  let shared = MTLResourceOptions::StorageModeShared;

  // GPU buffers (zero-copy unified memory)
  let real_buf = dev.new_buffer((total * 4) as u64, shared);
  let imag_buf = dev.new_buffer((total * 4) as u64, shared);
  let out_buf = dev.new_buffer((m * out_size * 4) as u64, shared);

  // Upload eigenvalues + bit-reverse table (small, one-time per config)
  let eig_buf = dev.new_buffer_with_data(
    sqrt_eigs.as_ptr() as *const _,
    (sqrt_eigs.len() * 4) as u64,
    shared,
  );
  let bit_rev = build_bit_reverse_table(traj_size);
  let rev_buf = dev.new_buffer_with_data(
    bit_rev.as_ptr() as *const _,
    (bit_rev.len() * 4) as u64,
    shared,
  );

  let seed: u32 = rand::Rng::random(&mut crate::simd_rng::rng());

  // Single command buffer for the entire pipeline
  let cmd = ctx.queue.new_command_buffer();
  let tg = MTLSize::new(256, 1, 1);
  let ts_u32 = traj_size as u32;

  // 1. Generate normals + scale + bit-reversal (GPU Philox RNG)
  {
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.gen_pso);
    enc.set_buffer(0, Some(&real_buf), 0);
    enc.set_buffer(1, Some(&imag_buf), 0);
    enc.set_buffer(2, Some(&eig_buf), 0);
    enc.set_buffer(3, Some(&rev_buf), 0);
    enc.set_bytes(4, 4, &ts_u32 as *const u32 as *const _);
    enc.set_bytes(5, 4, &seed as *const u32 as *const _);
    enc.dispatch_threads(MTLSize::new(total as u64, 1, 1), tg);
    enc.end_encoding();
  }

  // 2. FFT butterfly stages
  let grid_fft = MTLSize::new((total / 2) as u64, 1, 1);
  for stage in 0..log_n {
    let hs = (1u32 << stage) as u32;
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.butterfly_pso);
    enc.set_buffer(0, Some(&real_buf), 0);
    enc.set_buffer(1, Some(&imag_buf), 0);
    enc.set_bytes(2, 4, &ts_u32 as *const u32 as *const _);
    enc.set_bytes(3, 4, &hs as *const u32 as *const _);
    enc.dispatch_threads(grid_fft, tg);
    enc.end_encoding();
  }

  // 3. Extract
  {
    let os = out_size as u32;
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.extract_pso);
    enc.set_buffer(0, Some(&real_buf), 0);
    enc.set_buffer(1, Some(&out_buf), 0);
    enc.set_bytes(2, 4, &os as *const u32 as *const _);
    enc.set_bytes(3, 4, &ts_u32 as *const u32 as *const _);
    enc.set_bytes(4, 4, &scale as *const f32 as *const _);
    enc.dispatch_threads(MTLSize::new((m * out_size) as u64, 1, 1), tg);
    enc.end_encoding();
  }

  cmd.commit();
  cmd.wait_until_completed();

  // Zero-copy read from shared buffer
  let out_ptr = out_buf.contents() as *const f32;
  let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, m * out_size) };

  let fgn = arr2_f32::<T>(out_slice, m, out_size);
  if m == 1 {
    return Ok(Either::Left(fgn.row(0).to_owned()));
  }
  Ok(Either::Right(fgn))
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

impl<T: FloatExt, S: SeedExt> Fgn<T, S> {
  pub(crate) fn sample_metal_impl(&self, m: usize) -> Result<Either<Array1<T>, Array2<T>>> {
    let n = self.n;
    let offset = self.offset;
    let hurst = self.hurst.to_f64().unwrap();
    let t = self.t.unwrap_or(T::one()).to_f64().unwrap();
    let eigs: Vec<f32> = self
      .sqrt_eigenvalues
      .iter()
      .map(|x| x.to_f32().unwrap())
      .collect();
    sample_f32::<T>(&eigs, n, m, offset, hurst, t)
  }
}
