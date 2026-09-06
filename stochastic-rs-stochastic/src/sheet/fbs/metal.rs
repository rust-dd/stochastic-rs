//! # Metal sheets
//!
//! The two-dimensional circulant embedding on Metal: hashed complex Gaussian
//! noise scaled by the embedding's eigenvalue roots and scattered
//! bit-reversed along each row, the row transforms by the fGN pipeline's
//! butterfly, a transpose that bit-reverses the new rows, the column
//! transforms by the same butterfly, and a read-out that takes the real part
//! of the leading `m × n` block, subtracts its corner and adds Stein's linear
//! correction from two more hashed normals. One command buffer per chunk,
//! unified memory.

use metal::*;
use ndarray::Array2;
use parking_lot::Mutex;
use stochastic_rs_core::simd_rng::SeedExt;

use super::Fbs;
use super::SheetLaunch;
use crate::device::DeviceError;
use crate::noise::fgn::metal::MSL_COMMON;
use crate::noise::fgn::metal::build_bit_reverse_table;
use crate::traits::FloatExt;

type Result<T> = std::result::Result<T, DeviceError>;

const MSL_SHEET: &str = r#"
// One complex normal per embedding cell, scaled by the eigenvalue root and
// written to its row's bit-reversed slot; the hash runs on the batch-global
// cell so a chunk continues one launch's stream.
kernel void sheet_generate(
    device float* dst_real [[buffer(0)]],
    device float* dst_imag [[buffer(1)]],
    device const float* lam [[buffer(2)]],
    device const uint* bit_rev [[buffer(3)]],
    constant uint& cells [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& seed [[buffer(6)]],
    constant uint& first_cell [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint sheet = tid / cells;
    uint local = tid % cells;
    uint row = local / cols;
    uint col = local % cols;

    uint base = (tid + first_cell) * 4u + seed;
    float u1 = u01(pcg(base));
    float u2 = u01(pcg(base + 1u));
    float u3 = u01(pcg(base + 2u));
    float u4 = u01(pcg(base + 3u));
    float r_a = sqrt(-2.0f * log(u1 + 1e-10f));
    float r_b = sqrt(-2.0f * log(u3 + 1e-10f));
    float n_re = r_a * cos(6.28318530718f * u2);
    float n_im = r_b * cos(6.28318530718f * u4);

    float l = lam[local];
    uint dst = sheet * cells + row * cols + bit_rev[col];
    dst_real[dst] = n_re * l;
    dst_imag[dst] = n_im * l;
}

// Rows become columns, each new row bit-reversed for its own transform.
kernel void sheet_transpose(
    device const float* src_real [[buffer(0)]],
    device const float* src_imag [[buffer(1)]],
    device float* dst_real [[buffer(2)]],
    device float* dst_imag [[buffer(3)]],
    device const uint* bit_rev [[buffer(4)]],
    constant uint& cells [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint sheet = tid / cells;
    uint local = tid % cells;
    uint row = local / cols;
    uint col = local % cols;
    uint dst = sheet * cells + col * rows + bit_rev[row];
    dst_real[dst] = src_real[tid];
    dst_imag[dst] = src_imag[tid];
}

// The real part of the leading m × n block less its corner, plus Stein's
// linear correction sqrt(2 c2) · ((r (i+1)/m) z1 + (r (j+1)/n) z2) from two
// normals hashed on a counter past every cell of the batch.
kernel void sheet_extract(
    device const float* freq_real [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& cells [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant float& r [[buffer(6)]],
    constant float& corr [[buffer(7)]],
    constant uint& seed [[buffer(8)]],
    constant uint& corr_cell [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    uint per = m * n;
    uint sheet = tid / per;
    uint local = tid % per;
    uint i = local / n;
    uint j = local % n;
    uint base = sheet * cells;
    float value = freq_real[base + j * rows + i] - freq_real[base];

    uint hb = (corr_cell + sheet) * 4u + seed;
    float u1 = u01(pcg(hb));
    float u2 = u01(pcg(hb + 1u));
    float u3 = u01(pcg(hb + 2u));
    float u4 = u01(pcg(hb + 3u));
    float z1 = sqrt(-2.0f * log(u1 + 1e-10f)) * cos(6.28318530718f * u2);
    float z2 = sqrt(-2.0f * log(u3 + 1e-10f)) * cos(6.28318530718f * u4);
    float ty = r * float(i + 1u) / float(m);
    float tx = r * float(j + 1u) / float(n);
    output[tid] = value + corr * (ty * z1 + tx * z2);
}
"#;

#[derive(Clone)]
struct SheetCtx {
  ordinal: usize,
  device: Device,
  queue: CommandQueue,
  gen_pso: ComputePipelineState,
  butterfly_pso: ComputePipelineState,
  transpose_pso: ComputePipelineState,
  extract_pso: ComputePipelineState,
}

unsafe impl Send for SheetCtx {}

static CTX: Mutex<Option<SheetCtx>> = Mutex::new(None);

/// Per-configuration buffers, reused across same-size calls: the two complex
/// work buffers (one per transform direction), the output, the eigenvalue
/// roots and the two bit-reverse tables.
struct SizedSheet {
  real: Buffer,
  imag: Buffer,
  real_t: Buffer,
  imag_t: Buffer,
  out: Buffer,
  lam: Buffer,
  rev_cols: Buffer,
  rev_rows: Buffer,
  m: usize,
  n: usize,
  sheets: usize,
  key: (u64, u64),
}

unsafe impl Send for SizedSheet {}

/// The last [`crate::device::CACHE_SLOTS`] per-size states, least recent first.
static SIZED: Mutex<Vec<SizedSheet>> = Mutex::new(Vec::new());

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
  let source = format!("{MSL_COMMON}{MSL_SHEET}");
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

  let gen_pso = mk("sheet_generate")?;
  let butterfly_pso = mk("fft_butterfly")?;
  let transpose_pso = mk("sheet_transpose")?;
  let extract_pso = mk("sheet_extract")?;

  *g = Some(SheetCtx {
    ordinal,
    device,
    queue,
    gen_pso,
    butterfly_pso,
    transpose_pso,
    extract_pso,
  });
  Ok(())
}

/// The radix-2 stages of a batch of length-`n` transforms, on the buffer
/// pair the batch lies in.
fn encode_fft(cmd: &CommandBufferRef, ctx: &SheetCtx, real: &Buffer, imag: &Buffer, n: usize, total: usize) {
  let tg = MTLSize::new(256, 1, 1);
  let grid = MTLSize::new((total / 2) as u64, 1, 1);
  let n_u32 = n as u32;
  for stage in 0..n.trailing_zeros() {
    let hs = 1u32 << stage;
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.butterfly_pso);
    enc.set_buffer(0, Some(real), 0);
    enc.set_buffer(1, Some(imag), 0);
    enc.set_bytes(2, 4, &n_u32 as *const u32 as *const _);
    enc.set_bytes(3, 4, &hs as *const u32 as *const _);
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
  }
}

/// One chunk of the batch: `sheets` sheets from `first` on, as their
/// `sheets · m · n` values. `corr_cell` is the batch-global counter the
/// correction's normals hash from, past every cell of the whole batch.
fn sample_chunk(
  sheet: &SheetLaunch<'_, f32>,
  sheets: usize,
  first: usize,
  corr_cell: u32,
  seed: u32,
  ordinal: usize,
) -> Result<Vec<f32>> {
  let (m, n) = (sheet.m, sheet.n);
  let big_m = 2 * (m - 1);
  let big_n = 2 * (n - 1);
  let cells = big_m * big_n;
  let total = sheets * cells;
  let out_len = sheets * m * n;

  ensure_ctx(ordinal)?;
  // Clone the handles out of the global lock so another size can encode
  // concurrently; the per-size state below keeps its own lock for its buffers.
  let ctx = CTX.lock().as_ref().unwrap().clone();
  let dev = &ctx.device;
  let shared = MTLResourceOptions::StorageModeShared;

  let mut sized = SIZED.lock();
  let s = crate::device::lru_slot(
    &mut sized,
    |s| s.m == m && s.n == n && s.sheets == sheets && s.key == sheet.key,
    || {
      let floats = |len: usize| dev.new_buffer((len * 4) as u64, shared);
      let table = |rev: &[u32]| {
        dev.new_buffer_with_data(rev.as_ptr() as *const _, (rev.len() * 4) as u64, shared)
      };
      Ok(SizedSheet {
        real: floats(total),
        imag: floats(total),
        real_t: floats(total),
        imag_t: floats(total),
        out: floats(out_len),
        lam: dev.new_buffer_with_data(
          sheet.lam.as_ptr() as *const _,
          (sheet.lam.len() * 4) as u64,
          shared,
        ),
        rev_cols: table(&build_bit_reverse_table(big_n)),
        rev_rows: table(&build_bit_reverse_table(big_m)),
        m,
        n,
        sheets,
        key: sheet.key,
      })
    },
  )?;

  let cmd = ctx.queue.new_command_buffer();
  let tg = MTLSize::new(256, 1, 1);
  let cells_u32 = cells as u32;
  let rows_u32 = big_m as u32;
  let cols_u32 = big_n as u32;
  let first_cell = (first * cells) as u32;

  // 1. Draw, scale, scatter bit-reversed along the rows.
  {
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.gen_pso);
    enc.set_buffer(0, Some(&s.real), 0);
    enc.set_buffer(1, Some(&s.imag), 0);
    enc.set_buffer(2, Some(&s.lam), 0);
    enc.set_buffer(3, Some(&s.rev_cols), 0);
    enc.set_bytes(4, 4, &cells_u32 as *const u32 as *const _);
    enc.set_bytes(5, 4, &cols_u32 as *const u32 as *const _);
    enc.set_bytes(6, 4, &seed as *const u32 as *const _);
    enc.set_bytes(7, 4, &first_cell as *const u32 as *const _);
    enc.dispatch_threads(MTLSize::new(total as u64, 1, 1), tg);
    enc.end_encoding();
  }

  // 2. The row transforms.
  encode_fft(cmd, &ctx, &s.real, &s.imag, big_n, total);

  // 3. Transpose, bit-reversing the new rows.
  {
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.transpose_pso);
    enc.set_buffer(0, Some(&s.real), 0);
    enc.set_buffer(1, Some(&s.imag), 0);
    enc.set_buffer(2, Some(&s.real_t), 0);
    enc.set_buffer(3, Some(&s.imag_t), 0);
    enc.set_buffer(4, Some(&s.rev_rows), 0);
    enc.set_bytes(5, 4, &cells_u32 as *const u32 as *const _);
    enc.set_bytes(6, 4, &rows_u32 as *const u32 as *const _);
    enc.set_bytes(7, 4, &cols_u32 as *const u32 as *const _);
    enc.dispatch_threads(MTLSize::new(total as u64, 1, 1), tg);
    enc.end_encoding();
  }

  // 4. The column transforms.
  encode_fft(cmd, &ctx, &s.real_t, &s.imag_t, big_m, total);

  // 5. Read out the leading block, shifted and corrected.
  {
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.extract_pso);
    enc.set_buffer(0, Some(&s.real_t), 0);
    enc.set_buffer(1, Some(&s.out), 0);
    enc.set_bytes(2, 4, &cells_u32 as *const u32 as *const _);
    enc.set_bytes(3, 4, &rows_u32 as *const u32 as *const _);
    enc.set_bytes(4, 4, &m_u32 as *const u32 as *const _);
    enc.set_bytes(5, 4, &n_u32 as *const u32 as *const _);
    enc.set_bytes(6, 4, &sheet.r as *const f32 as *const _);
    enc.set_bytes(7, 4, &sheet.corr as *const f32 as *const _);
    enc.set_bytes(8, 4, &seed as *const u32 as *const _);
    enc.set_bytes(9, 4, &corr_cell as *const u32 as *const _);
    enc.dispatch_threads(MTLSize::new(out_len as u64, 1, 1), tg);
    enc.end_encoding();
  }

  cmd.commit();
  cmd.wait_until_completed();

  // Shared storage: the pointer is the same memory the GPU wrote.
  let out_ptr = s.out.contents() as *const f32;
  Ok(unsafe { std::slice::from_raw_parts(out_ptr, out_len) }.to_vec())
}

impl<T: FloatExt, S: SeedExt, B> Fbs<T, S, B> {
  /// `sheets` sheets on the selected Metal device, in chunks that fit the
  /// batch budget: one seed for the whole batch, the cell hash offset per
  /// chunk by the cells already produced and the correction's hash by the
  /// sheets, so the result is the same whatever the budget.
  pub(crate) fn sample_metal_sheets(
    &self,
    sheets: usize,
    device: &crate::device::Metal,
  ) -> Result<Vec<Array2<T>>> {
    let (m, n) = (self.m, self.n);
    let cells = self.cells();
    let lam: Vec<f32> = self.lam.iter().map(|x| x.to_f32().unwrap()).collect();
    let launch = SheetLaunch {
      lam: &lam,
      m,
      n,
      r: self.r.to_f32().unwrap(),
      corr: self.correction().to_f32().unwrap(),
      key: self.launch_key(),
    };
    let seed = self.seed.seed_value() as u32;
    let rows = crate::device::chunk_rows(device.batch_budget, 4 * cells + m * n, 4);
    let mut out = Vec::with_capacity(sheets);
    let mut first = 0;
    while first < sheets {
      let len = rows.min(sheets - first);
      let corr_cell = (sheets * cells + first) as u32;
      let flat = sample_chunk(&launch, len, first, corr_cell, seed, device.ordinal)?;
      out.extend(self.sheets_from_flat(&flat));
      first += len;
    }
    Ok(out)
  }
}

#[cfg(test)]
mod chunk_tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::device::Metal;
  use crate::traits::ProcessExt;

  /// A batch produced in chunks equals one launch, sheet for sheet: one seed
  /// per batch, a cell offset per chunk and a sheet offset for the
  /// correction's normals.
  #[test]
  fn chunks_are_bit_identical_to_one_launch() {
    let fbs = |device: Metal| {
      Fbs::<f32, _>::new(0.7, 9, 5, 1.0, Deterministic::new(5)).with_backend(device)
    };
    let whole = fbs(Metal::default()).sample_par(7);
    // Two sheets per chunk: four launches for seven sheets.
    let per_sheet = (4 * 4 * 8 * 4 + 9 * 5) * 4;
    let chunked = fbs(Metal::default().with_batch_budget(2 * per_sheet)).sample_par(7);
    assert_eq!(whole, chunked);
    assert_ne!(whole[0], whole[1]);
  }

  /// The process's own seed drives its device sheets: two sheets built from
  /// the same `Deterministic` seed agree, a different seed differs.
  #[test]
  fn a_sheet_honours_its_own_seed_on_the_device() {
    let fbs = |seed: u64| Fbs::<f32, _>::new(0.7, 9, 9, 1.0, Deterministic::new(seed)).on::<Metal>();
    assert_eq!(fbs(3).sample_par(3), fbs(3).sample_par(3));
    assert_ne!(fbs(3).sample(), fbs(4).sample());
    assert_eq!(fbs(3).sample(), fbs(3).sample());
  }
}
