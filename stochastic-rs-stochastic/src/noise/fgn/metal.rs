//! # Metal GPU
//!
//! macOS Metal compute backend for Fgn sampling: keyed draw + Box-Muller ->
//! eigenvalue scale -> bit-reversal -> FFT butterfly stages -> read-out, all
//! in one command buffer over unified memory.
//!
//! The transform is memory-bound, so what shapes the pipeline is how many
//! times the batch crosses the memory system, not the arithmetic. A butterfly
//! stage of its own reads and writes the whole batch — measured on an M4 Max
//! at ten thousand paths over four thousand points, one stage moves 1.31 GB
//! in 3.9 ms, 340 GB/s, and a stage that only copies the same bytes costs the
//! same, so the twiddles and the Box-Muller are free next to the traffic.
//! Thirteen such stages were 49 ms of a 57 ms launch. A contiguous block of
//! 2^k bit-reversed positions is closed under the first k stages, so
//! [`MSL_FGN`]'s `gen_tile_fft` draws its own block, permutes it and runs
//! those k stages inside threadgroup memory, writing the batch once instead
//! of k + 1 times; the last stage exports the scaled real part in place of
//! the transform, which removes the read-out pass too. The butterflies are
//! the same butterflies in the same order, so the output is bit-identical to
//! the stage-per-dispatch pipeline this replaced — measured over 61 million
//! values across three shapes, not assumed. Where the cut falls depends on
//! the transform length and on how wide a threadgroup the compiled kernel
//! admits, so both shapes it can take are swept against the embedding's own
//! covariance by `tests/fgn_metal_tile_cuts.rs`.
use metal::*;
use ndarray::Array2;
use parking_lot::Mutex;
use stochastic_rs_core::simd_rng::SeedExt;

use super::Fgn;
use crate::device::DeviceError;

type Result<T> = std::result::Result<T, DeviceError>;

/// What every MSL pipeline of this crate's FFT family starts from: the
/// uniform, the 64-bit keyed draw behind it, and the radix-2 butterfly stage.
/// The sheet pipeline concatenates its own kernels behind it.
pub(crate) const MSL_COMMON: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float u01(uint x) {
    return (float(x >> 8) + 0.5f) / 16777216.0f;
}

// SplitMix64. The counter these pipelines draw on is an element of the whole
// batch, and a batch of a million 512-point paths already passes 2^30 of
// them, where a 32-bit counter hands two chunks the same stream.
inline ulong sm64(ulong x) {
    x += 0x9e3779b97f4a7c15UL;
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9UL;
    x ^= x >> 27; x *= 0x94d049bb133111ebUL;
    x ^= x >> 31;
    return x;
}

// The four uniforms of one cell: two 64-bit mixes of the cell number under
// the batch's seed, each read as two words.
inline float4 u01x4(ulong cell, uint seed) {
    ulong k = cell ^ ((ulong)seed * 0x9e3779b97f4a7c15UL);
    ulong h1 = sm64(k);
    ulong h2 = sm64(k ^ 0xd1b54a32d192ed03UL);
    return float4(u01((uint)h1), u01((uint)(h1 >> 32)),
                  u01((uint)h2), u01((uint)(h2 >> 32)));
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
"#;

/// The fGN kernels proper: the head of the pipeline, and its tail.
const MSL_FGN: &str = r#"
// The draw, the eigenvalue scale, the bit-reversal and the first `stages`
// butterfly stages, all inside one threadgroup's tile. Each thread produces
// two (re, im) pairs from the four uniforms of a cell of the batch, fed into
// two Box-Muller transforms, then owns one butterfly of every stage.
//
// The value belonging at bit-reversed position p is the draw of the natural
// index rev(p) — the permutation is its own inverse — so the tile can start
// from its own positions rather than from a scatter across the batch.
kernel void gen_tile_fft(
    device float* real [[buffer(0)]],
    device float* imag [[buffer(1)]],
    device const float* sqrt_eigs [[buffer(2)]],
    device const uint* bit_rev [[buffer(3)]],
    constant uint& traj_size [[buffer(4)]],
    constant uint& seed [[buffer(5)]],
    constant ulong& first_cell [[buffer(6)]],
    constant uint& stages [[buffer(7)]],
    threadgroup float* sh_re [[threadgroup(0)]],
    threadgroup float* sh_im [[threadgroup(1)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint half_tile [[threads_per_threadgroup]])
{
    uint tile = half_tile * 2;
    uint tile_base = gid * tile;
    uint batch = tile_base / traj_size;
    uint within = tile_base % traj_size;

    for (uint k = lid; k < tile; k += half_tile) {
        uint nat = bit_rev[within + k];
        float4 u = u01x4((ulong)(batch * traj_size + nat) + first_cell, seed);
        float r_a = sqrt(-2.0f * log(u.x + 1e-10f));
        float r_b = sqrt(-2.0f * log(u.z + 1e-10f));
        float eig = sqrt_eigs[nat];
        sh_re[k] = r_a * cos(6.28318530718f * u.y) * eig;
        sh_im[k] = r_b * cos(6.28318530718f * u.w) * eig;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0; s < stages; ++s) {
        uint half_stride = 1u << s;
        uint stride = half_stride * 2;
        uint group = lid / half_stride;
        uint pos = lid % half_stride;
        uint i = group * stride + pos;
        uint j = i + half_stride;

        float angle = -2.0f * 3.14159265358979323846f * float(pos) / float(stride);
        float tw_r = cos(angle);
        float tw_i = sin(angle);

        float tr = sh_re[j] * tw_r - sh_im[j] * tw_i;
        float ti = sh_re[j] * tw_i + sh_im[j] * tw_r;
        float ar = sh_re[i];
        float ai = sh_im[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sh_re[i] = ar + tr;
        sh_im[i] = ai + ti;
        sh_re[j] = ar - tr;
        sh_im[j] = ai - ti;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint k = lid; k < tile; k += half_tile) {
        real[tile_base + k] = sh_re[k];
        imag[tile_base + k] = sh_im[k];
    }
}

// The last butterfly stage, exporting the read-out in place of the
// transform. Only the real half leaves, so the imaginary output is never
// formed and `imag[i]` is never read: three loads and one store where the
// stage that writes the whole transform costs four and four, and the pass
// that used to copy the read-out out of it disappears.
kernel void fft_butterfly_extract(
    device const float* real [[buffer(0)]],
    device const float* imag [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& half_stride [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& out_size [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    uint butterflies_per_batch = n / 2;
    uint batch = tid / butterflies_per_batch;
    uint pos = tid % butterflies_per_batch;
    uint i = batch * n + pos;
    uint j = i + half_stride;

    float angle = -2.0f * 3.14159265358979323846f * float(pos) / float(half_stride * 2);
    float tw_r = cos(angle);
    float tw_i = sin(angle);
    float tr = real[j] * tw_r - imag[j] * tw_i;
    float ar = real[i];

    // Position `l` of the transform is entry `l - 1` of the read-out.
    uint out_base = batch * out_size;
    if (pos >= 1 && pos - 1 < out_size) {
        output[out_base + pos - 1] = (ar + tr) * scale;
    }
    uint l = pos + half_stride;
    if (l - 1 < out_size) {
        output[out_base + l - 1] = (ar - tr) * scale;
    }
}
"#;

#[derive(Clone)]
struct MetalCtx {
  ordinal: usize,
  device: Device,
  queue: CommandQueue,
  tile_pso: ComputePipelineState,
  butterfly_pso: ComputePipelineState,
  readout_pso: ComputePipelineState,
}

unsafe impl Send for MetalCtx {}

static CTX: Mutex<Option<MetalCtx>> = Mutex::new(None);

/// Per-configuration GPU buffers, reused across same-size calls. Re-allocating
/// the trajectory buffers and re-uploading the eigenvalue / bit-reverse tables
/// on every call was the dominant per-call cost; the gen kernel overwrites
/// `real`/`imag` each time, so reuse is safe.
struct SizedMetal {
  real_buf: Buffer,
  imag_buf: Buffer,
  out_buf: Buffer,
  eig_buf: Buffer,
  rev_buf: Buffer,
  n: usize,
  m: usize,
  offset: usize,
  hurst_bits: u64,
  t_bits: u64,
}

unsafe impl Send for SizedMetal {}

/// The last [`crate::device::CACHE_SLOTS`] per-size states, least recent first.
static SIZED: Mutex<Vec<SizedMetal>> = Mutex::new(Vec::new());

fn ensure_ctx(ordinal: usize) -> Result<()> {
  let mut g = CTX.lock();
  if g.as_ref().is_some_and(|c| c.ordinal == ordinal) {
    return Ok(());
  }
  // A new device invalidates the per-size buffers of the old one.
  *g = None;
  SIZED.lock().clear();
  let device = crate::euler::metal::metal_device(ordinal)?;
  let queue = device.new_command_queue();
  let source = format!("{MSL_COMMON}{MSL_FGN}");
  let lib = device
    .new_library_with_source(&source, &CompileOptions::new())
    .map_err(|e| DeviceError::Compile(format!("MSL compile: {e}")))?;

  let mk = |name: &str| -> Result<ComputePipelineState> {
    let f = lib
      .get_function(name, None)
      .map_err(|e| DeviceError::Launch(format!("get {name}: {e}")))?;
    device
      .new_compute_pipeline_state_with_function(&f)
      .map_err(|e| DeviceError::Launch(format!("{name} PSO: {e}")))
  };

  let tile_pso = mk("gen_tile_fft")?;
  let butterfly_pso = mk("fft_butterfly")?;
  let readout_pso = mk("fft_butterfly_extract")?;

  *g = Some(MetalCtx {
    ordinal,
    device,
    queue,
    tile_pso,
    butterfly_pso,
    readout_pso,
  });
  Ok(())
}

/// How the tile pass is cut for a transform of `2^log_n` points: the tile's
/// log-size and how many butterfly stages fit inside it.
///
/// The tile is one threadgroup's worth of threadgroup memory (two `f32`
/// arrays of `2^tile_log`) and one butterfly per thread, so its half is
/// bounded by what the compiled kernel admits per threadgroup — 1024 on the
/// Apple parts measured, which puts the tile at 2048 points and 16 KB. A
/// device that admits fewer just runs more of the stages as their own
/// dispatch; the output does not change with the cut. The last stage always
/// stays outside, because that is the one that exports the read-out.
fn tile_plan(log_n: usize, max_threads: usize) -> (usize, usize) {
  let mut tile_log = log_n.min(11);
  while tile_log > 1 && (1usize << (tile_log - 1)) > max_threads {
    tile_log -= 1;
  }
  (tile_log, tile_log.min(log_n - 1))
}

/// Where each index of a length-`n` transform lands when its bits are
/// reversed, the order a decimation-in-time butterfly wants its input in.
pub(crate) fn build_bit_reverse_table(n: usize) -> Vec<u32> {
  let log_n = n.trailing_zeros() as usize;
  let bits = usize::BITS as usize;
  (0..n)
    .map(|i| (i.reverse_bits() >> (bits - log_n)) as u32)
    .collect()
}

/// The pipeline itself: leaves the increments in the device buffer it wrote
/// them to and hands that buffer over, so a consumer on the same device — the
/// Euler engine — reads them without a round trip through host memory.
/// Returns the buffer and the row length.
pub(crate) fn sample_f32_buffer(
  sqrt_eigs: &[f32],
  n: usize,
  m: usize,
  offset: usize,
  hurst: f64,
  t: f64,
  seed: u32,
  first_cell: u64,
  ordinal: usize,
) -> Result<(Buffer, usize)> {
  let traj_size = 2 * n;
  let out_size = n - offset;
  let scale = (out_size.max(1) as f32).powf(-(hurst as f32)) * (t as f32).powf(hurst as f32);
  let total = m * traj_size;
  let log_n = traj_size.trailing_zeros() as usize;

  ensure_ctx(ordinal)?;
  // Clone the handles out of the global lock so another size can encode
  // concurrently; the per-size state below keeps its own lock for its buffers.
  let ctx = CTX.lock().as_ref().unwrap().clone();
  let dev = &ctx.device;
  let shared = MTLResourceOptions::StorageModeShared;
  let hb = hurst.to_bits();
  let tb = t.to_bits();

  // Allocate trajectory buffers + upload the eigenvalue / bit-reverse tables
  // once per configuration; reuse across same-size calls.
  let mut sized = SIZED.lock();
  let s = crate::device::lru_slot(
    &mut sized,
    |s| s.n == n && s.m == m && s.offset == offset && s.hurst_bits == hb && s.t_bits == tb,
    || {
      let real_buf = dev.new_buffer((total * 4) as u64, shared);
      let imag_buf = dev.new_buffer((total * 4) as u64, shared);
      let out_buf = dev.new_buffer((m * out_size * 4) as u64, shared);
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
      Ok(SizedMetal {
        real_buf,
        imag_buf,
        out_buf,
        eig_buf,
        rev_buf,
        n,
        m,
        offset,
        hurst_bits: hb,
        t_bits: tb,
      })
    },
  )?;

  // Single command buffer for the entire pipeline
  let cmd = ctx.queue.new_command_buffer();
  let tg = MTLSize::new(256, 1, 1);
  let ts_u32 = traj_size as u32;
  let (tile_log, tile_stages) = tile_plan(
    log_n,
    ctx.tile_pso.max_total_threads_per_threadgroup() as usize,
  );
  let tile = 1usize << tile_log;

  // 1. Draw + eigenvalue scale + bit-reversal + the stages the tile covers
  {
    let st = tile_stages as u32;
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.tile_pso);
    enc.set_buffer(0, Some(&s.real_buf), 0);
    enc.set_buffer(1, Some(&s.imag_buf), 0);
    enc.set_buffer(2, Some(&s.eig_buf), 0);
    enc.set_buffer(3, Some(&s.rev_buf), 0);
    enc.set_bytes(4, 4, &ts_u32 as *const u32 as *const _);
    enc.set_bytes(5, 4, &seed as *const u32 as *const _);
    enc.set_bytes(6, 8, &first_cell as *const u64 as *const _);
    enc.set_bytes(7, 4, &st as *const u32 as *const _);
    enc.set_threadgroup_memory_length(0, (tile * 4) as u64);
    enc.set_threadgroup_memory_length(1, (tile * 4) as u64);
    enc.dispatch_thread_groups(
      MTLSize::new((total / tile) as u64, 1, 1),
      MTLSize::new((tile / 2) as u64, 1, 1),
    );
    enc.end_encoding();
  }

  // 2. The stages between the tile and the last one, a dispatch each
  let grid_fft = MTLSize::new((total / 2) as u64, 1, 1);
  for stage in tile_stages..log_n - 1 {
    let hs = 1u32 << stage;
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.butterfly_pso);
    enc.set_buffer(0, Some(&s.real_buf), 0);
    enc.set_buffer(1, Some(&s.imag_buf), 0);
    enc.set_bytes(2, 4, &ts_u32 as *const u32 as *const _);
    enc.set_bytes(3, 4, &hs as *const u32 as *const _);
    enc.dispatch_threads(grid_fft, tg);
    enc.end_encoding();
  }

  // 3. The last stage, which writes the read-out rather than the transform
  {
    let hs = 1u32 << (log_n - 1);
    let os = out_size as u32;
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.readout_pso);
    enc.set_buffer(0, Some(&s.real_buf), 0);
    enc.set_buffer(1, Some(&s.imag_buf), 0);
    enc.set_bytes(2, 4, &ts_u32 as *const u32 as *const _);
    enc.set_bytes(3, 4, &hs as *const u32 as *const _);
    enc.set_buffer(4, Some(&s.out_buf), 0);
    enc.set_bytes(5, 4, &os as *const u32 as *const _);
    enc.set_bytes(6, 4, &scale as *const f32 as *const _);
    enc.dispatch_threads(grid_fft, tg);
    enc.end_encoding();
  }

  cmd.commit();
  cmd.wait_until_completed();

  Ok((s.out_buf.clone(), out_size))
}

/// What one launch of a chunked batch needs: the row that starts it and how
/// many rows it takes.
struct Chunk {
  first: usize,
  len: usize,
}

/// The rows of `m` paths cut into launches that fit the device's batch
/// budget. One seed serves the whole batch and each chunk carries the count
/// of elements already produced, so the result is the same whatever the
/// budget — the property `chunk_tests` pins.
fn chunks(fgn_n: usize, out_size: usize, m: usize, device: &crate::device::Metal) -> Vec<Chunk> {
  let budget = device
    .batch_budget
    .min(crate::euler::metal::working_set(device.ordinal));
  let rows = crate::device::chunk_rows(budget, 4 * fgn_n + out_size, 4);
  let mut first = 0;
  let mut out = Vec::new();
  while first < m {
    let len = rows.min(m - first);
    out.push(Chunk { first, len });
    first += len;
  }
  out
}

/// The Metal kernels compute in single precision, so the entry points below
/// are single-precision too: an `Fgn<f64>` on this backend is a compile
/// error rather than an `f32` computation behind an `f64` type.
impl<S: SeedExt, B> Fgn<f32, S, B> {
  /// `m` paths on the selected Metal device, read back into an array.
  pub(crate) fn sample_metal_impl<S2: SeedExt>(
    &self,
    m: usize,
    seed_src: &S2,
    device: &crate::device::Metal,
  ) -> Result<Array2<f32>> {
    let out_size = self.n - self.offset;
    let mut out = Array2::<f32>::zeros((m, out_size));
    self.over_metal_chunks(m, seed_src, device, |chunk, rows| {
      let mut dst = out.slice_mut(ndarray::s![chunk.first..chunk.first + chunk.len, ..]);
      dst
        .as_slice_mut()
        .expect("contiguous rows")
        .copy_from_slice(rows);
      Ok(())
    })?;
    Ok(out)
  }

  /// `f` over the rows of `m` paths, read where the device wrote them.
  ///
  /// Unified memory makes the launch's own output buffer readable as it
  /// stands, so a caller that only folds the batch never needs it copied
  /// into an owned array first — at ten thousand paths over four thousand
  /// points that copy is a hundred and fifty-six megabytes and a sixth of
  /// the call.
  pub(crate) fn map_metal_impl<S2: SeedExt, R: Send>(
    &self,
    m: usize,
    seed_src: &S2,
    device: &crate::device::Metal,
    f: impl Fn(ndarray::ArrayView1<f32>) -> R + Sync,
  ) -> Result<Vec<R>> {
    use rayon::prelude::*;
    let out_size = (self.n - self.offset).max(1);
    let mut out = Vec::with_capacity(m);
    self.over_metal_chunks(m, seed_src, device, |_, rows| {
      out.par_extend(
        rows
          .par_chunks(out_size)
          .map(|row| f(ndarray::ArrayView1::from(row))),
      );
      Ok(())
    })?;
    Ok(out)
  }

  /// Runs the pipeline chunk by chunk, handing `consume` the rows of each
  /// launch as the device left them. The rows are consumed before the next
  /// launch, which reuses the same cached buffer behind them.
  fn over_metal_chunks<S2: SeedExt>(
    &self,
    m: usize,
    seed_src: &S2,
    device: &crate::device::Metal,
    mut consume: impl FnMut(&Chunk, &[f32]) -> Result<()>,
  ) -> Result<()> {
    let (n, offset) = (self.n, self.offset);
    let out_size = n - offset;
    let hurst = self.hurst as f64;
    let t = self.t.unwrap_or(1.0) as f64;
    let eigs = self
      .sqrt_eigenvalues
      .as_slice()
      .expect("the eigenvalues are contiguous");
    let seed = seed_src.seed_value() as u32;
    for chunk in chunks(n, out_size, m, device) {
      let (buf, cols) = sample_f32_buffer(
        eigs,
        n,
        chunk.len,
        offset,
        hurst,
        t,
        seed,
        (chunk.first * 2 * n) as u64,
        device.ordinal,
      )?;
      // SAFETY: shared storage the launch has finished writing into, of at
      // least `chunk.len * cols` floats by construction.
      let rows =
        unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, chunk.len * cols) };
      consume(&chunk, rows)?;
    }
    Ok(())
  }
}

#[cfg(test)]
mod chunk_tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::device::Metal;
  use crate::traits::ProcessExt;

  /// A batch produced in chunks equals one launch, path for path: one seed per
  /// batch and an element offset per chunk.
  #[test]
  fn chunks_are_bit_identical_to_one_launch() {
    let fgn = |device: Metal| {
      Fgn::<f32, _>::new(0.7, 512, Some(1.0), Deterministic::new(5)).with_backend(device)
    };
    let whole = fgn(Metal::default()).sample_par(9);
    // Two paths per chunk: five launches for nine paths.
    let chunked = fgn(Metal::default().with_batch_budget((4 * 512 + 512) * 4 * 2)).sample_par(9);
    assert_eq!(whole, chunked);
    assert_ne!(whole[0], whole[1]);
  }

  /// A wrapper's own seed drives its device paths: two `Fbm`s built from the
  /// same `Deterministic` seed agree, a different seed differs, and the inner
  /// `Unseeded` fGN never enters.
  #[test]
  fn fbm_honours_its_own_seed_on_the_device() {
    use crate::process::fbm::Fbm;
    let fbm =
      |seed: u64| Fbm::<f32, _>::new(0.7, 256, Some(1.0), Deterministic::new(seed)).on::<Metal>();
    assert_eq!(fbm(3).sample_par(3), fbm(3).sample_par(3));
    assert_ne!(fbm(3).sample_par(1), fbm(4).sample_par(1));
    assert_eq!(fbm(3).sample(), fbm(3).sample());
  }
}
