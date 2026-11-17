//! cudarc + NVRTC Euler engine: one thread per path, in `f32` or `f64` after `T`, on the Metal
//! kernels' Philox counters, so the two back-ends agree seed for seed up to libm rounding.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::*;
use cudarc::nvrtc;
use ndarray::Array2;
use ndarray::Array3;
use parking_lot::Mutex;

use super::EulerCoefficients;
use super::EulerKernel;
use super::EulerSpec;
use super::FgnSpec;
use super::families;
use super::kernel::Shape;
use super::launch_scalars;
use super::philox;
use crate::device::Cuda;
use crate::device::DeviceError;
use crate::device::DeviceInfo;
use crate::noise::fgn::cuda::PinnedHost;
use crate::traits::FloatExt;

mod launch;
mod pipeline;
mod profile;

use launch::device_paths;
use pipeline::pipelined_paths;
pub use profile::KernelProfile;
pub use profile::kernel_profile;

type Result<T> = std::result::Result<T, DeviceError>;

/// Which cached output buffer a precision uses. Implemented for the two the
/// kernels are rendered in; `cudarc`'s slices are typed, so the cache is one
/// array per precision rather than one of bytes.
trait CachedOut: DeviceRepr + ValidAsZeroBits + Copy {
  fn slots(kernels: &mut Kernels) -> &mut [Option<CudaSlice<Self>>; 2];

  fn staging(kernels: &mut Kernels) -> &mut [Option<PinnedHost<Self>>; 2];
}

impl CachedOut for f32 {
  fn slots(kernels: &mut Kernels) -> &mut [Option<CudaSlice<f32>>; 2] {
    &mut kernels.out_f32
  }

  fn staging(kernels: &mut Kernels) -> &mut [Option<PinnedHost<f32>>; 2] {
    &mut kernels.staging_f32
  }
}

impl CachedOut for f64 {
  fn slots(kernels: &mut Kernels) -> &mut [Option<CudaSlice<f64>>; 2] {
    &mut kernels.out_f64
  }

  fn staging(kernels: &mut Kernels) -> &mut [Option<PinnedHost<f64>>; 2] {
    &mut kernels.staging_f64
  }
}

/// The cached buffer of `slot`, grown to `len` if it is smaller, taken out of
/// the cache for the caller to launch into. [`return_output`] puts it back.
fn take_output<R: CachedOut>(
  kernels: &mut Kernels,
  stream: &Arc<CudaStream>,
  slot: usize,
  len: usize,
) -> Result<CudaSlice<R>> {
  match R::slots(kernels)[slot].take() {
    Some(buffer) if buffer.len() >= len => Ok(buffer),
    // The old buffer is released *before* the larger one is asked for: held
    // across the allocation, a batch that doubles needs both at once, and
    // the peak is what fails first on a card that is nearly full.
    held => {
      drop(held);
      stream
        .alloc_zeros::<R>(len)
        .map_err(|e| driver_error("alloc out", e))
    }
  }
}

/// Puts a launch's output buffer back for the next call to grow into.
fn return_output<R: CachedOut>(kernels: &mut Kernels, slot: usize, buffer: CudaSlice<R>) {
  R::slots(kernels)[slot] = Some(buffer);
}

/// The cached host landing buffer, grown to `len` if it is smaller, taken out
/// of the cache for the caller to copy into. [`return_staging`] puts it back.
fn take_staging<R: CachedOut>(
  kernels: &mut Kernels,
  slot: usize,
  len: usize,
) -> Result<PinnedHost<R>> {
  match R::staging(kernels)[slot].take() {
    Some(buffer) if buffer.len() >= len => Ok(buffer),
    // Freed before the larger one is pinned, for the reason `take_output`
    // gives: page-locked memory is the scarcer of the two.
    held => {
      drop(held);
      PinnedHost::<R>::alloc(len)
    }
  }
}

/// Puts the host landing buffer back for the next call to grow into.
fn return_staging<R: CachedOut>(kernels: &mut Kernels, slot: usize, buffer: PinnedHost<R>) {
  R::staging(kernels)[slot] = Some(buffer);
}

/// A driver failure as a [`DeviceError`], with the one code a caller can act
/// on kept apart: an allocation that ran out of memory is retried by the
/// engine's batch loop with a smaller chunk, where every other code is the
/// same however the batch is cut.
fn driver_error(what: &str, e: DriverError) -> DeviceError {
  if e.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY {
    DeviceError::OutOfMemory(format!("{what}: {e}"))
  } else {
    DeviceError::Launch(format!("{what}: {e}"))
  }
}

/// Threads per block. The kernel carries a path's whole state — the four
/// components, the 512-slot history block, the program stack — in registers
/// and local memory, so occupancy is bounded by register pressure long before
/// it is bounded by threads. `LaunchConfig::for_num_elems` asks for 1024,
/// which on that footprint spills or is refused outright by the driver; 256
/// is the largest block that leaves every family room.
const BLOCK: u32 = 256;

/// One thread per path, in blocks of [`BLOCK`].
fn paths_config(paths: u32) -> LaunchConfig {
  LaunchConfig {
    grid_dim: (paths.div_ceil(BLOCK), 1, 1),
    block_dim: (BLOCK, 1, 1),
    shared_mem_bytes: 0,
  }
}

/// The `float` / `double` kernel: the launch header around the C frame body
/// in [`super::kernel`].
const CUDA_HEADER: &str = r#"extern "C" __global__ void euler_paths_REAL(
    REAL* __restrict__ out,
    const REAL* __restrict__ params,
    unsigned int family, unsigned int components, unsigned int noises,
    REAL x00, REAL x01, REAL x02, REAL x03,
    REAL dt, REAL sqrt_dt,
    unsigned int seed_lo, unsigned int seed_hi, unsigned int steps, unsigned int paths,
    unsigned int first_path,
    const REAL* __restrict__ incs, unsigned int increments,
    const REAL* __restrict__ curve, unsigned int n_curves,
    REAL jump_lambda, unsigned int has_jumps,
    unsigned int jump_law, REAL jump_a, REAL jump_b, REAL jump_c,
    unsigned int step_first,
    unsigned int gamma_law, REAL g1_shape, REAL g1_scale, REAL g1_per,
    REAL g2_shape, REAL g2_scale, REAL g2_per,
    const REAL* __restrict__ lift_decay, const REAL* __restrict__ lift_weight,
    const REAL* __restrict__ lift_drift_scale, unsigned int has_lift, unsigned int lift_n,
    REAL lift_db, REAL lift_fb, REAL lift_x0, unsigned int hist_slot,
    unsigned int series_n, unsigned int series_live, unsigned int table_n, REAL table_u0,
    const REAL* __restrict__ program, unsigned int program_n)
{
    unsigned int path = blockIdx.x * blockDim.x + threadIdx.x;
    const REAL x0[4] = { x00, x01, x02, x03 };
"#;

fn kernel_source(real: &'static str, shape: Shape) -> String {
  let lang = super::kernel::cuda_language(real);
  let prelude = super::kernel::prelude(&lang);
  let body = super::kernel::render_for(&lang, shape);
  format!("{prelude}{}{body}}}\n", CUDA_HEADER.replace("REAL", real))
}

struct Kernels {
  ordinal: usize,
  context: Arc<CudaContext>,
  stream: Arc<CudaStream>,
  /// Second stream of the batch pipeline: chunk `k + 1` computes on one
  /// while chunk `k` copies back on the other.
  stream_b: Arc<CudaStream>,
  /// The launches' output buffers, kept between calls: slot 0 for a single
  /// launch, slots 0 and 1 for the two-stream pipeline.
  ///
  /// `alloc_zeros` per launch both allocates and clears; on Metal the same
  /// allocation was the single largest cost of a batch — three to four times
  /// the kernel at a hundred thousand paths — and there is no reason CUDA
  /// pays it either. The kernel writes every element it is asked for, so a
  /// reused buffer needs no clearing.
  out_f32: [Option<CudaSlice<f32>>; 2],
  out_f64: [Option<CudaSlice<f64>>; 2],

  /// The page-locked host buffer a single launch lands in: pageable memory crosses the bus at about
  /// 3.6 GB/s. It holds the largest batch seen, as the device buffers do, until `Cuda::release`.
  staging_f32: [Option<PinnedHost<f32>>; 2],
  staging_f64: [Option<PinnedHost<f64>>; 2],

  /// One compiled kernel per launch shape and precision, built on first use.
  ///
  /// A T4 reports what the monolithic body costs: 80 registers and 3552 bytes
  /// of local memory per thread in `f32`, 130 and 7104 in `f64` — three and
  /// one block per multiprocessor. A kernel rendered for one family carries
  /// that family's step alone and declares only the scratch it reaches, which
  /// is where the occupancy comes back.
  functions: HashMap<(Shape, &'static str), CudaFunction>,
}

/// SAFETY: the raw driver handles inside are only ever reached through this
/// mutex, and the two streams are ordered against each other by the explicit
/// synchronise each chunk's drain performs before its slot is reused.
unsafe impl Send for Kernels {}

static KERNELS: Mutex<Option<Kernels>> = Mutex::new(None);

/// Drops the engine's kernels, output buffers and pinned staging cached for `ordinal`.
pub(crate) fn release(ordinal: usize) {
  let mut kernels = KERNELS.lock();
  if kernels.as_ref().is_some_and(|k| k.ordinal == ordinal) {
    *kernels = None;
  }
}

#[cfg(test)]
pub(crate) fn cached_context() -> Option<std::sync::Weak<CudaContext>> {
  KERNELS.lock().as_ref().map(|k| Arc::downgrade(&k.context))
}

/// What a launch of one compiled shape needs, cloned under the lock that compiled it.
struct Launcher {
  context: Arc<CudaContext>,
  streams: [Arc<CudaStream>; 2],
  func: CudaFunction,
}

/// The cache that compiled `context`'s launch, or the error a release or rebind on another thread
/// leaves behind: a launch never takes buffers from, or returns them to, another context's cache.
fn lender<'a>(
  kernels: &'a mut Option<Kernels>,
  context: &Arc<CudaContext>,
) -> Result<&'a mut Kernels> {
  kernels
    .as_mut()
    .filter(|k| Arc::ptr_eq(&k.context, context))
    .ok_or_else(|| {
      DeviceError::Launch(
        "the kernel cache was released or rebound to another device mid-launch".into(),
      )
    })
}

/// The CUDA device at `ordinal`, or why it cannot be used.
pub(crate) fn probe(ordinal: usize) -> Result<DeviceInfo> {
  let ctx =
    CudaContext::new(ordinal).map_err(|e| DeviceError::Unavailable(format!("CudaContext: {e}")))?;
  let name = ctx
    .name()
    .map_err(|e| DeviceError::Unavailable(format!("device name: {e}")))?;
  Ok(DeviceInfo::new(
    "Cuda",
    name,
    &["f32", "f64"],
    Some(ordinal),
  ))
}

/// Opens the device at `ordinal` if it is not already open, compiles the kernel for `shape` at
/// `real` if that one is not already built, and hands back what a launch of it needs.
fn ensure_kernels(ordinal: usize, shape: Shape, real: &'static str) -> Result<Launcher> {
  let mut guard = KERNELS.lock();
  let kernels = match &mut *guard {
    Some(kernels) if kernels.ordinal == ordinal => kernels,
    slot => slot.insert(open(ordinal)?),
  };
  let func = match kernels.functions.entry((shape, real)) {
    std::collections::hash_map::Entry::Occupied(entry) => entry.get().clone(),
    std::collections::hash_map::Entry::Vacant(entry) => {
      let name = format!("euler_paths_{real}");
      let ptx = nvrtc::compile_ptx(kernel_source(real, shape))
        .map_err(|e| DeviceError::Compile(format!("NVRTC {name}: {e}")))?;
      let module = kernels
        .context
        .load_module(ptx)
        .map_err(|e| DeviceError::Launch(format!("load {name}: {e}")))?;
      let function = module
        .load_function(&name)
        .map_err(|e| DeviceError::Launch(format!("fn {name}: {e}")))?;
      entry.insert(function).clone()
    }
  };
  Ok(Launcher {
    context: kernels.context.clone(),
    streams: [kernels.stream.clone(), kernels.stream_b.clone()],
    func,
  })
}

/// The device at `ordinal` with its two streams and nothing compiled or cached yet.
fn open(ordinal: usize) -> Result<Kernels> {
  let ctx =
    CudaContext::new(ordinal).map_err(|e| DeviceError::Unavailable(format!("CudaContext: {e}")))?;
  let stream = ctx
    .new_stream()
    .map_err(|e| DeviceError::Launch(format!("stream: {e}")))?;
  let stream_b = ctx
    .new_stream()
    .map_err(|e| DeviceError::Launch(format!("stream: {e}")))?;
  Ok(Kernels {
    ordinal,
    context: stream.context().clone(),
    stream,
    stream_b,
    out_f32: [None, None],
    out_f64: [None, None],
    staging_f32: [None, None],
    staging_f64: [None, None],
    functions: HashMap::new(),
  })
}

impl<T: FloatExt> EulerKernel<T> for Cuda {
  fn euler_kernel<P: EulerCoefficients<T>>(
    &self,
    process: &P,
    first: usize,
    m: usize,
    seed: u64,
  ) -> Result<Array2<T>> {
    device_paths(
      self.ordinal,
      process.euler_spec(),
      process.initial_state(),
      process.grid_points(),
      process.time_step(),
      first,
      m,
      seed,
      process.fgn_spec(),
      process.curves(),
      process.jump_intensity(),
      process.jump_sizes(),
      process.step_first(),
      process.gamma_draws(),
      process.lift_spec(),
      process.series_terms(),
      process.table_spec(),
      process.program_spec(),
      // A family may step more components than the process reports, so the
      // first plane is taken by the count the launch actually wrote.
      |data, _| {
        Array2::from_shape_vec(
          (m, process.grid_points()),
          data[..m * process.grid_points()].to_vec(),
        )
        .expect("the kernel writes m * n values in its first plane")
      },
    )
  }

  /// The launch's rows lent to `f` rather than copied out.
  ///
  /// The page-locked landing buffer is what `f` reads, so the batch crosses
  /// the bus once and is not copied again on this side. That second copy is
  /// the larger of the two on a host with slow pages: forty megabytes of
  /// fresh `Vec` is ten thousand first-touch faults before a byte moves.
  fn euler_kernel_lend<P: EulerCoefficients<T>, O>(
    &self,
    process: &P,
    first: usize,
    m: usize,
    seed: u64,
    f: impl FnOnce(ndarray::ArrayView2<T>) -> O,
  ) -> Result<O> {
    let n = process.grid_points();
    device_paths(
      self.ordinal,
      process.euler_spec(),
      process.initial_state(),
      n,
      process.time_step(),
      first,
      m,
      seed,
      process.fgn_spec(),
      process.curves(),
      process.jump_intensity(),
      process.jump_sizes(),
      process.step_first(),
      process.gamma_draws(),
      process.lift_spec(),
      process.series_terms(),
      process.table_spec(),
      process.program_spec(),
      |data, _| {
        f(ndarray::ArrayView2::from_shape((m, n), &data[..m * n])
          .expect("the kernel writes m * n values in its first plane"))
      },
    )
  }

  /// A system's launch: the same kernel, its state slots filled from the
  /// process's own initial state and every component's plane returned.
  fn euler_system_kernel<const D: usize, P: super::EulerSystem<T, D>>(
    &self,
    process: &P,
    first: usize,
    m: usize,
    seed: u64,
  ) -> Result<Array3<T>> {
    let spec = process.euler_spec();
    super::check_arity(&spec, D);
    let slots = process.initial_state();
    device_paths(
      self.ordinal,
      spec,
      slots,
      process.grid_points(),
      process.time_step(),
      first,
      m,
      seed,
      process.fgn_spec(),
      process.curves(),
      process.jump_intensity(),
      process.jump_sizes(),
      process.step_first(),
      process.gamma_draws(),
      process.lift_spec(),
      process.series_terms(),
      process.table_spec(),
      process.program_spec(),
      // `check_arity` allows a family to step *more* components than the
      // process returns, so the array is shaped by what the launch wrote —
      // not by `D`, which is only the count the caller will read back out.
      |data, planes| {
        Array3::from_shape_vec((planes, m, process.grid_points()), data.to_vec())
          .expect("the kernel returns components * m * n values")
      },
    )
  }

  fn batch_budget(&self) -> usize {
    self.batch_budget
  }

  /// The output is indexed in 64 bits, a path and its increment row `path * increments` in an
  /// `unsigned int`, and the fGN pipeline a fractional launch runs first in an `int`.
  fn index_rows(
    &self,
    rows: usize,
    _spec: EulerSpec<T>,
    _steps: usize,
    fgn: Option<FgnSpec<'_, T>>,
  ) -> usize {
    let streams = fgn.as_ref().map_or(1, |f| f.streams);
    let rows = crate::device::rows_within_index_limit(rows, streams, u32::MAX as usize);
    match fgn {
      Some(f) => crate::noise::fgn::cuda::sampler::index_rows(rows, f.streams, f.n),
      None => rows,
    }
  }

  /// Chunks alternate between two streams: while chunk `k` copies back
  /// through pinned memory, chunk `k + 1` is already computing.
  fn euler_kernel_batch<P: EulerCoefficients<T>>(
    &self,
    process: &P,
    m: usize,
    seed: u64,
  ) -> Result<Array2<T>> {
    super::check_path_index(m)?;
    let n = process.grid_points();
    let rows = super::launch_rows(self, n, process.euler_spec(), n, process.fgn_spec());
    if m <= rows {
      return self.euler_kernel(process, 0, m, seed);
    }
    // The pipelined batch draws its own normals and runs no fGN pipeline or lift, so it would step
    // Brownian motion in their place: such a process takes the single launch chunk by chunk.
    if process.fgn_spec().is_some() || process.lift_spec().is_some() {
      let mut out = Array2::<T>::zeros((m, n));
      // Through `over_chunks`, not a plain loop: a chunk that runs the card
      // out of memory is retried at half the size, which is the whole point
      // of that helper and was unwired on the back-end it was written for.
      crate::device::over_chunks(m, rows, |first, len| {
        let chunk = self.euler_kernel(process, first, len, seed)?;
        out
          .slice_mut(ndarray::s![first..first + len, ..])
          .assign(&chunk);
        Ok(())
      })?;
      return Ok(out);
    }
    let planes = pipelined_paths(
      self.ordinal,
      process.euler_spec(),
      process.initial_state(),
      n,
      process.time_step(),
      m,
      rows,
      seed,
      process.curves(),
      process.jump_intensity(),
      process.jump_sizes(),
      process.step_first(),
      process.gamma_draws(),
      None,
      process.series_terms(),
      process.table_spec(),
      process.program_spec(),
    )?;
    Ok(planes.index_axis_move(ndarray::Axis(0), 0))
  }
}
