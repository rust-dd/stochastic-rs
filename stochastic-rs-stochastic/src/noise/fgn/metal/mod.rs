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
//! Thirteen such stages were 49 ms of a 57 ms launch. Three things cut that
//! down, and all three are about passes rather than arithmetic:
//!
//! - A contiguous block of 2^k bit-reversed positions is closed under the
//!   first k stages, so `gen_tile_fft` draws its own block, permutes it and
//!   runs those k stages inside threadgroup memory — the batch is written
//!   once instead of k + 1 times.
//! - The stages above the tile go two to a dispatch: the quadruple a thread
//!   of `fft_radix4` owns is closed under both, so a pair of stages is one
//!   crossing rather than two.
//! - The last pair exports the read-out in place of the transform, and
//!   exports *both* halves of it. The real and imaginary parts of a
//!   circulant-embedding transform are two independent paths of the same law
//!   (Dietrich & Newsam 1997), so one transform serves two rows of the batch
//!   and the pass that used to copy the read-out out disappears.
//!
//! Together, fifteen passes over the batch became three, over half as many
//! transforms: 2.5x at ten thousand paths over four thousand points, on top
//! of the 4x the tile alone was worth.
//!
//! **What this does not preserve is the values.** The pairing changes which
//! transform a row comes from, and holding two stages in registers lets the
//! compiler contract the multiply-add that spans them, so the device's output
//! is a different realisation of the same law than earlier releases produced.
//! A device's stream has never been reproducible across versions — see
//! [`FgnBackend`](crate::device::FgnBackend)'s table — and the host backends
//! are untouched. What *is* pinned, by `tests/fgn_metal_pipeline.rs`, is that
//! every row is the transform it claims to be, entry by entry, against a
//! direct Fourier sum over the same keyed draw; that the two rows of a pair
//! are independent; and that a batch cut anywhere still equals one launch.

mod kernels;

use kernels::MSL_FGN;
use metal::*;
use ndarray::Array2;
use parking_lot::Mutex;
use stochastic_rs_core::simd_rng::SeedExt;

pub(crate) use self::kernels::MSL_COMMON;
use super::Fgn;
use crate::device::DeviceError;

type Result<T> = std::result::Result<T, DeviceError>;

#[derive(Clone)]
struct MetalCtx {
  ordinal: usize,
  device: Device,
  queue: CommandQueue,
  tile_pso: ComputePipelineState,
  radix4_pso: ComputePipelineState,
  readout_pso: ComputePipelineState,
  readout2_pso: ComputePipelineState,
}

unsafe impl Send for MetalCtx {}

static CTX: Mutex<Option<MetalCtx>> = Mutex::new(None);

/// Per-configuration GPU buffers, reused across same-size calls. Re-allocating
/// the trajectory buffers and re-uploading the eigenvalue / bit-reverse tables
/// on every call was the dominant per-call cost; the gen kernel overwrites
/// `real`/`imag` each time, so reuse is safe.
///
/// The read-out's destination is deliberately *not* here. These four are only
/// ever touched while [`SIZED`] is held, but the read-out buffer outlives the
/// call — its whole purpose is to be read afterwards, and on the engine's
/// fractional path it is read by a second launch — so a cache entry shared
/// across threads would let one thread's pipeline overwrite the increments
/// another thread is still consuming. It comes from [`OUT_POOL`] instead,
/// which hands a buffer to one holder at a time.
struct SizedMetal {
  real_buf: Buffer,
  imag_buf: Buffer,
  eig_buf: Buffer,
  rev_buf: Buffer,
  n: usize,
  m: usize,
  offset: usize,
  parity: u32,
  hurst_bits: u64,
  t_bits: u64,
}

unsafe impl Send for SizedMetal {}

/// The last [`crate::device::CACHE_SLOTS`] per-size states, least recent first.
static SIZED: Mutex<Vec<SizedMetal>> = Mutex::new(Vec::new());

/// Read-out buffers nobody is holding, longest first.
///
/// A fresh `StorageModeShared` buffer has to be faulted in page by page before
/// anything can be read out of it, and at ten thousand paths over four
/// thousand points that is a hundred and fifty-six megabytes: measured on an
/// M4 Max, allocating one per launch costs 34 % of `Fou::sample_par` and 59 %
/// of `Fgn::sample_par` at that size. So they are pooled rather than
/// reallocated — but pooled by *ownership*, one holder at a time, which is
/// what keying them by size got wrong.
static OUT_POOL: Mutex<Vec<Buffer>> = Mutex::new(Vec::new());

/// A read-out buffer of at least `bytes`, taken out of [`OUT_POOL`] so no
/// other caller can be handed the same one.
fn take_out_buffer(dev: &Device, bytes: u64) -> Buffer {
  let mut pool = OUT_POOL.lock();
  match pool.iter().position(|b| b.length() >= bytes) {
    Some(i) => pool.remove(i),
    None => dev.new_buffer(bytes.max(4), MTLResourceOptions::StorageModeShared),
  }
}

/// Hands a read-out buffer back once its holder is done with it, keeping the
/// [`crate::device::CACHE_SLOTS`] longest.
fn return_out_buffer(buf: Buffer) {
  let mut pool = OUT_POOL.lock();
  let at = pool
    .iter()
    .position(|b| b.length() < buf.length())
    .unwrap_or(pool.len());
  pool.insert(at, buf);
  pool.truncate(crate::device::CACHE_SLOTS);
}

std::thread_local! {
  /// The read-out buffer [`sample_f32_buffer`] last handed this thread.
  ///
  /// That entry point returns the buffer to a caller who keeps it — the Euler
  /// engine binds it into a launch of its own — so nothing inside this module
  /// can see the borrow end. What can be said is that the borrow is over by
  /// the time the *same* thread asks for another: every consumer reads the
  /// buffer within the call that obtained it (`EulerKernel::euler_kernel`
  /// waits for the launch that reads the increments before it returns), so a
  /// thread back here has finished with the last one. Reclaiming it then is
  /// what keeps a Monte-Carlo loop on one thread reusing a single buffer,
  /// while two threads simply hold two.
  static LENT: std::cell::RefCell<Option<Buffer>> = const { std::cell::RefCell::new(None) };
}

/// The compile options every library here is built with.
///
/// Metal's relaxed float mode is the default, and a default is not a promise:
/// it has moved across OS versions, and `MTLCompileOptions` replaced the flag
/// with a three-way mode in Metal 3.1. Setting it explicitly is what keeps a
/// seed reproducing the same path after a system update — the values these
/// kernels produce depend on it, and nothing else in the crate pins it.
fn compile_options() -> CompileOptions {
  let options = CompileOptions::new();
  options.set_fast_math_enabled(true);
  options
}

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
    .new_library_with_source(&source, &compile_options())
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
  let radix4_pso = mk("fft_radix4")?;
  let readout_pso = mk("fft_radix4_extract")?;
  let readout2_pso = mk("fft_butterfly_extract")?;

  *g = Some(MetalCtx {
    ordinal,
    device,
    queue,
    tile_pso,
    radix4_pso,
    readout_pso,
    readout2_pso,
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
/// dispatch. The last stage always stays outside, because that is the one
/// that exports the read-out, and the count left outside is kept even so
/// each of them is half of a radix-4 pair — a stage shifted into the tile
/// costs a threadgroup round trip, a stage left over costs a batch one.
fn tile_plan(log_n: usize, max_threads: usize) -> (usize, usize) {
  let mut tile_log = log_n.min(11);
  while tile_log > 1 && (1usize << (tile_log - 1)) > max_threads {
    tile_log -= 1;
  }
  let mut stages = tile_log.min(log_n - 1);
  if (log_n - stages) % 2 == 1 && stages >= 1 {
    stages -= 1;
  }
  (tile_log, stages)
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

/// Everything one launch needs to know, so the two entry points below differ
/// only in who ends up holding the read-out.
#[derive(Clone, Copy)]
struct Launch<'a> {
  sqrt_eigs: &'a [f32],
  n: usize,
  m: usize,
  offset: usize,
  hurst: f64,
  t: f64,
  seed: u32,
  first_cell: u64,
  ordinal: usize,
}

impl Launch<'_> {
  /// Floats the read-out writes, which is what a buffer for it must hold.
  fn out_len(&self) -> usize {
    self.m * (self.n - self.offset)
  }
}

/// The pipeline itself, writing its read-out into `out`.
///
/// `first_cell` names where the launch sits in the batch: it is the absolute
/// index of its first row times `2n`, which is what every caller already
/// passes. One transform serves *two* rows — the real and imaginary halves of
/// a circulant-embedding transform are two independent paths of the same law
/// (Dietrich & Newsam 1997) — so a launch whose first row is odd computes the
/// transform in front of it as well and drops the half it does not own. Which
/// values a row gets is therefore a function of its absolute index alone, and
/// a batch cut anywhere still equals one launch.
fn run_pipeline(launch: Launch<'_>, out: &Buffer) -> Result<usize> {
  let Launch {
    sqrt_eigs,
    n,
    m,
    offset,
    hurst,
    t,
    seed,
    first_cell,
    ordinal,
  } = launch;
  let traj_size = 2 * n;
  let out_size = n - offset;
  let scale = (out_size.max(1) as f32).powf(-(hurst as f32)) * (t as f32).powf(hurst as f32);
  let log_n = traj_size.trailing_zeros() as usize;
  let parity = ((first_cell / traj_size as u64) & 1) as u32;
  let base_cell = first_cell - u64::from(parity) * traj_size as u64;
  let transforms = (parity as usize + m).div_ceil(2);
  let total = transforms * traj_size;

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
    |s| {
      s.n == n
        && s.m == m
        && s.offset == offset
        && s.parity == parity
        && s.hurst_bits == hb
        && s.t_bits == tb
    },
    || {
      let real_buf = dev.new_buffer((total * 4) as u64, shared);
      let imag_buf = dev.new_buffer((total * 4) as u64, shared);
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
        eig_buf,
        rev_buf,
        n,
        m,
        offset,
        parity,
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
    enc.set_bytes(6, 8, &base_cell as *const u64 as *const _);
    enc.set_bytes(7, 4, &st as *const u32 as *const _);
    enc.set_threadgroup_memory_length(0, (tile * 4) as u64);
    enc.set_threadgroup_memory_length(1, (tile * 4) as u64);
    enc.dispatch_thread_groups(
      MTLSize::new((total / tile) as u64, 1, 1),
      MTLSize::new((tile / 2) as u64, 1, 1),
    );
    enc.end_encoding();
  }

  // 2. The stages between the tile and the last pair, two to a dispatch
  for stage in (tile_stages..log_n.saturating_sub(2)).step_by(2) {
    let s0 = stage as u32;
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.radix4_pso);
    enc.set_buffer(0, Some(&s.real_buf), 0);
    enc.set_buffer(1, Some(&s.imag_buf), 0);
    enc.set_bytes(2, 4, &ts_u32 as *const u32 as *const _);
    enc.set_bytes(3, 4, &s0 as *const u32 as *const _);
    enc.dispatch_threads(MTLSize::new((total / 4) as u64, 1, 1), tg);
    enc.end_encoding();
  }

  // 3. The last stages, which export the read-out rather than the transform.
  // `tile_plan` leaves an even count outside the tile wherever it can, so the
  // pair is the usual tail and the single stage only serves a four-point
  // transform.
  {
    let single = log_n - tile_stages == 1;
    let os = out_size as u32;
    let rows = m as u32;
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(if single {
      &ctx.readout2_pso
    } else {
      &ctx.readout_pso
    });
    enc.set_buffer(0, Some(&s.real_buf), 0);
    enc.set_buffer(1, Some(&s.imag_buf), 0);
    enc.set_bytes(2, 4, &ts_u32 as *const u32 as *const _);
    enc.set_buffer(3, Some(out), 0);
    enc.set_bytes(4, 4, &os as *const u32 as *const _);
    enc.set_bytes(5, 4, &scale as *const f32 as *const _);
    enc.set_bytes(6, 4, &rows as *const u32 as *const _);
    enc.set_bytes(7, 4, &parity as *const u32 as *const _);
    let lanes = if single { total / 2 } else { total / 4 };
    enc.dispatch_threads(MTLSize::new(lanes as u64, 1, 1), tg);
    enc.end_encoding();
  }

  cmd.commit();
  cmd.wait_until_completed();

  Ok(out_size)
}

/// One launch whose read-out the caller keeps: the buffer comes back with it,
/// so a consumer on the same device — the Euler engine, which binds it into a
/// launch of its own — reads the increments without a round trip through host
/// memory. Returns the buffer and the row length.
///
/// The buffer is the caller's alone until this thread asks for another (see
/// [`LENT`]), which is what makes the hand-over safe to hold across a second
/// launch.
#[allow(clippy::too_many_arguments)]
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
  let launch = Launch {
    sqrt_eigs,
    n,
    m,
    offset,
    hurst,
    t,
    seed,
    first_cell,
    ordinal,
  };
  ensure_ctx(ordinal)?;
  let dev = CTX.lock().as_ref().unwrap().device.clone();
  if let Some(done) = LENT.with_borrow_mut(Option::take) {
    return_out_buffer(done);
  }
  let out = take_out_buffer(&dev, (launch.out_len() * 4) as u64);
  let out_size = run_pipeline(launch, &out)?;
  LENT.with_borrow_mut(|slot| *slot = Some(out.clone()));
  Ok((out, out_size))
}

/// One launch whose read-out `consume` reads where it lies, the buffer going
/// back to the pool the moment it returns.
///
/// The borrow is bracketed here, so this path needs none of
/// [`sample_f32_buffer`]'s deferred reclamation and holds no buffer between
/// calls.
fn with_f32_rows<R>(launch: Launch<'_>, consume: impl FnOnce(&[f32]) -> Result<R>) -> Result<R> {
  ensure_ctx(launch.ordinal)?;
  let dev = CTX.lock().as_ref().unwrap().device.clone();
  let len = launch.out_len();
  let out = take_out_buffer(&dev, (len * 4) as u64);
  let result = run_pipeline(launch, &out).and_then(|_| {
    // SAFETY: shared storage the launch has finished writing into, of at
    // least `len` floats by construction, and held by nobody else.
    consume(unsafe { std::slice::from_raw_parts(out.contents() as *const f32, len) })
  });
  return_out_buffer(out);
  result
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
  // A row costs half a transform's `real`/`imag` — two paths come out of one
  // — plus its own output row.
  let rows = crate::device::chunk_rows(budget, 2 * fgn_n + out_size, 4);
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
  /// launch as the device left them. The read-out is borrowed for exactly as
  /// long as `consume` runs, so no other launch — on this thread or another —
  /// can be writing the rows it is reading.
  fn over_metal_chunks<S2: SeedExt>(
    &self,
    m: usize,
    seed_src: &S2,
    device: &crate::device::Metal,
    mut consume: impl FnMut(&Chunk, &[f32]) -> Result<()>,
  ) -> Result<()> {
    let (n, offset) = (self.n, self.offset);
    let out_size = n - offset;
    let eigs = self
      .sqrt_eigenvalues
      .as_slice()
      .expect("the eigenvalues are contiguous");
    let seed = seed_src.seed_value() as u32;
    for chunk in chunks(n, out_size, m, device) {
      let launch = Launch {
        sqrt_eigs: eigs,
        n,
        m: chunk.len,
        offset,
        hurst: self.hurst as f64,
        t: self.t.unwrap_or(1.0) as f64,
        seed,
        first_cell: (chunk.first * 2 * n) as u64,
        ordinal: device.ordinal,
      };
      with_f32_rows(launch, |rows| consume(&chunk, rows))?;
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
