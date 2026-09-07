//! cudarc + NVRTC device path of the Euler engine: one CUDA thread per path,
//! the whole recursion in the kernel, normals from the same counter hash of
//! `(path, step, seed)` as the Metal kernel (so the two device back-ends
//! agree seed for seed up to libm rounding), in `f32` or `f64`
//! according to `T` — NVIDIA hardware has native double precision.

use std::any::TypeId;
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
use super::Reduce;
use super::kernel::Shape;
use crate::device::Cuda;
use crate::device::DeviceError;
use crate::device::DeviceInfo;
use crate::noise::fgn::cuda::PinnedHost;
use crate::traits::FloatExt;

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
  let held = R::slots(kernels)[slot].take();
  match held {
    Some(buffer) if buffer.len() >= len => Ok(buffer),
    _ => stream
      .alloc_zeros::<R>(len)
      .map_err(|e| driver_error("alloc out", e)),
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
    _ => PinnedHost::<R>::alloc(len),
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

/// The `float` / `double` kernel: the launch header around the body the
/// Metal back-end renders too ([`super::kernel`]).
const CUDA_HEADER: &str = r#"extern "C" __global__ void euler_paths_REAL(
    REAL* __restrict__ out,
    const REAL* __restrict__ params,
    unsigned int family, unsigned int components, unsigned int noises,
    REAL x00, REAL x01, REAL x02, REAL x03,
    REAL dt, REAL sqrt_dt,
    unsigned int seed, unsigned int steps, unsigned int paths,
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

  /// The host buffer a single launch lands in, page-locked and kept between
  /// calls for the same reason the device buffers are.
  ///
  /// `clone_dtoh` allocates a fresh `Vec` and copies into pageable memory,
  /// which the driver has to stage through a bounce buffer of its own: about
  /// 3.6 GB/s where a page-locked destination reaches the bus. The batch pays
  /// that on every launch, along with the first-touch page fault of every
  /// page of the fresh allocation — tens of thousands of them at a hundred
  /// thousand paths. The pipelined path already stages through pinned memory;
  /// this is the same for the single launch, which is what every batch that
  /// fits the budget takes.
  ///
  /// It holds the largest batch seen so far, page-locked, until the process
  /// ends — the same bargain the device buffers strike, and the same size as
  /// the `Vec` the call returns, so the peak is what it always was.
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

/// SAFETY: every device operation is serialised through the one stream.
unsafe impl Send for Kernels {}

static KERNELS: Mutex<Option<Kernels>> = Mutex::new(None);

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

/// What the driver reports about the compiled engine kernel.
///
/// Registers and local memory per thread are what bound occupancy, and a
/// kernel rendered for one launch shape is the engine's answer to both: it
/// carries that family's step alone and declares only the scratch it can
/// reach, where the monolithic body carried every family's step and every
/// optional frame block's scratch whether the launch used them or not. These
/// numbers say whether that worked on a given card, and they cannot be read
/// on a machine without the device — which is why they are printed by an
/// example rather than assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelProfile {
  /// Registers per thread.
  pub registers: i32,
  /// Local (per-thread, off-chip) memory in bytes.
  pub local_bytes: i32,
  /// Statically allocated shared memory in bytes.
  pub shared_bytes: i32,
  /// The largest block the driver will launch this kernel with.
  pub max_threads_per_block: i32,
  /// Concurrent blocks per multiprocessor at [`block`](Self::block) threads.
  pub blocks_per_multiprocessor: u32,
  /// The block size the occupancy above was computed for.
  pub block: u32,
  /// Threads one multiprocessor of this device can hold resident, read from
  /// the device rather than assumed.
  ///
  /// It is the denominator of [`occupancy_percent`](Self::occupancy_percent)
  /// and it is not a constant across architectures: Turing holds 1024 where
  /// most generations hold 2048, so a hard-coded 2048 reports a T4's full
  /// multiprocessor as half empty.
  pub threads_per_multiprocessor: i32,
}

impl KernelProfile {
  /// The share of a multiprocessor's thread slots the kernel fills at
  /// [`block`](Self::block) threads, in percent — 100 meaning the device
  /// cannot hold another warp of this kernel.
  pub fn occupancy_percent(&self) -> u32 {
    let slots = self.threads_per_multiprocessor.max(1) as u32;
    self.blocks_per_multiprocessor * self.block * 100 / slots
  }
}

/// [`KernelProfile`] for the kernel a plain diffusion compiles at `real` —
/// the shape every specialised launch is measured against — on the device at
/// `ordinal`, at a block size of `block` threads.
pub fn kernel_profile(ordinal: usize, real: &'static str, block: u32) -> Result<KernelProfile> {
  let shape = Shape::new(super::families::Family::GeometricBrownian, false, false);
  ensure_kernels(ordinal, shape, real)?;
  let guard = KERNELS.lock();
  let kernels = guard.as_ref().expect("initialised");
  let func = kernels
    .functions
    .get(&(shape, real))
    .expect("compiled for this shape");
  let attr = |what: &str, v: std::result::Result<i32, DriverError>| -> Result<i32> {
    v.map_err(|e| driver_error(what, e))
  };
  Ok(KernelProfile {
    registers: attr("num_regs", func.num_regs())?,
    local_bytes: attr("local_size_bytes", func.local_size_bytes())?,
    shared_bytes: attr("shared_size_bytes", func.shared_size_bytes())?,
    max_threads_per_block: attr("max_threads_per_block", func.max_threads_per_block())?,
    blocks_per_multiprocessor: func
      .occupancy_max_active_blocks_per_multiprocessor(block, 0, None)
      .map_err(|e| driver_error("occupancy", e))?,
    block,
    threads_per_multiprocessor: attr(
      "max_threads_per_multiprocessor",
      kernels.context.attribute(
        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
      ),
    )?,
  })
}

/// Opens the device at `ordinal` if it is not already open, and compiles the
/// kernel for `shape` at `real` if that one is not already built.
fn ensure_kernels(ordinal: usize, shape: Shape, real: &'static str) -> Result<()> {
  let mut guard = KERNELS.lock();
  if !guard.as_ref().is_some_and(|k| k.ordinal == ordinal) {
    let ctx = CudaContext::new(ordinal)
      .map_err(|e| DeviceError::Unavailable(format!("CudaContext: {e}")))?;
    let stream = ctx
      .new_stream()
      .map_err(|e| DeviceError::Launch(format!("stream: {e}")))?;
    let context = stream.context();
    let stream_b = ctx
      .new_stream()
      .map_err(|e| DeviceError::Launch(format!("stream: {e}")))?;
    *guard = Some(Kernels {
      ordinal,
      context: context.clone(),
      stream,
      stream_b,
      out_f32: [None, None],
      out_f64: [None, None],
      staging_f32: [None, None],
      staging_f64: [None, None],
      functions: HashMap::new(),
    });
  }
  let kernels = guard.as_mut().expect("initialised");
  if kernels.functions.contains_key(&(shape, real)) {
    return Ok(());
  }
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
  kernels.functions.insert((shape, real), function);
  Ok(())
}

/// One launch, with `finish` run over the values it wrote.
///
/// The finisher is what lets a caller that only reads the batch — a mapped
/// fold, say — never see a copy of it: the page-locked landing buffer is lent
/// for the call and taken back after, where a caller that wants to own the
/// values copies them itself. On a host with slow pages that copy is the
/// larger part of the wall, bigger than the bus crossing it follows.
#[allow(clippy::too_many_arguments)]
fn run<R, O>(
  ordinal: usize,
  real: &'static str,
  params: [R; crate::euler::PARAM_SLOTS],
  x0: [R; 4],
  dt: R,
  family: u32,
  components: u32,
  noises: u32,
  seed: u32,
  first: usize,
  n: usize,
  m: usize,
  increments: Option<(&CudaSlice<R>, u32)>,
  curve: &[R],
  n_curves: u32,
  jump_lambda: R,
  use_jumps: u32,
  jump_law: u32,
  jump_a: R,
  jump_b: R,
  jump_c: R,
  step_first: u32,
  gamma_law: u32,
  g1_shape: R,
  g1_scale: R,
  g1_per: R,
  g2_shape: R,
  g2_scale: R,
  g2_per: R,
  lift: [&[R]; 3],
  has_lift: u32,
  lift_n: u32,
  lift_db: R,
  lift_fb: R,
  lift_x0: R,
  hist_slot: u32,
  series_n: u32,
  series_live: u32,
  table_n: u32,
  table_u0: R,
  program: &[R],
  program_n: u32,
  reduce: Reduce,
  finish: impl FnOnce(&[R]) -> O,
) -> Result<O>
where
  R: DeviceRepr + ValidAsZeroBits + Copy + num_traits::Float + CachedOut,
{
  let shape = Shape::new(
    super::families::Family::from_code(family).expect("a declared family"),
    use_jumps != 0 || jump_law != 0,
    gamma_law != 0,
  )
  .with_reduce(reduce);
  ensure_kernels(ordinal, shape, real)?;
  let out_len = components as usize * m * reduce.stride(n);
  let mut guard = KERNELS.lock();
  let kernels = guard.as_mut().expect("initialised");
  let stream = kernels.stream.clone();
  let func = kernels
    .functions
    .get(&(shape, real))
    .expect("compiled for this shape")
    .clone();
  let mut d_out = take_output::<R>(kernels, &stream, 0, out_len)?;
  drop(guard);
  let stream = &stream;
  let func = &func;
  let d_params = stream
    .clone_htod(&params[..])
    .map_err(|e| driver_error("htod params", e))?;
  let sqrt_dt = dt.sqrt();
  let (steps, paths, first_path) = (n as u32, m as u32, first as u32);
  // The kernel always binds the increment pointer; an unused slot gets one
  // element rather than a null. A supplied slice was written on this device by
  // the fGN pipeline and is bound where it lies.
  let use_incs = increments.map_or(0, |(_, streams)| streams);
  let owned;
  let d_incs = match increments {
    Some((slice, _)) => slice,
    None => {
      owned = stream
        .alloc_zeros::<R>(1)
        .map_err(|e| driver_error("alloc incs", e))?;
      &owned
    }
  };
  // The kernel always binds the curve pointer; an unused slot gets one
  // element rather than a null.
  let use_curve = n_curves;
  // The three lift tables; an unused table gets one element rather than a
  // null, as the increment and curve pointers do.
  let mut d_lift = Vec::with_capacity(3);
  for table in lift {
    d_lift.push(if table.is_empty() {
      stream
        .alloc_zeros::<R>(1)
        .map_err(|e| driver_error("alloc lift", e))?
    } else {
      stream
        .clone_htod(table)
        .map_err(|e| driver_error("htod lift", e))?
    });
  }
  let d_curve = if curve.is_empty() {
    stream
      .alloc_zeros::<R>(1)
      .map_err(|e| driver_error("alloc curve", e))?
  } else {
    stream
      .clone_htod(curve)
      .map_err(|e| driver_error("htod curve", e))?
  };
  let d_program = stream
    .clone_htod(program)
    .map_err(|e| driver_error("htod program", e))?;
  unsafe {
    stream
      .launch_builder(func)
      .arg(&mut d_out)
      .arg(&d_params)
      .arg(&family)
      .arg(&components)
      .arg(&noises)
      .arg(&x0[0])
      .arg(&x0[1])
      .arg(&x0[2])
      .arg(&x0[3])
      .arg(&dt)
      .arg(&sqrt_dt)
      .arg(&seed)
      .arg(&steps)
      .arg(&paths)
      .arg(&first_path)
      .arg(d_incs)
      .arg(&use_incs)
      .arg(&d_curve)
      .arg(&use_curve)
      .arg(&jump_lambda)
      .arg(&use_jumps)
      .arg(&jump_law)
      .arg(&jump_a)
      .arg(&jump_b)
      .arg(&jump_c)
      .arg(&step_first)
      .arg(&gamma_law)
      .arg(&g1_shape)
      .arg(&g1_scale)
      .arg(&g1_per)
      .arg(&g2_shape)
      .arg(&g2_scale)
      .arg(&g2_per)
      .arg(&d_lift[0])
      .arg(&d_lift[1])
      .arg(&d_lift[2])
      .arg(&has_lift)
      .arg(&lift_n)
      .arg(&lift_db)
      .arg(&lift_fb)
      .arg(&lift_x0)
      .arg(&hist_slot)
      .arg(&series_n)
      .arg(&series_live)
      .arg(&table_n)
      .arg(&table_u0)
      .arg(&d_program)
      .arg(&program_n)
      .launch(paths_config(paths))
      .map_err(|e| DeviceError::Launch(format!("euler_paths: {e}")))?;
  }
  // Only this launch's own rows: the buffer may be larger, having been grown
  // by an earlier one. The copy goes to page-locked memory and the `Vec` is
  // filled from there — a plain `clone_dtoh` would allocate that `Vec` per
  // launch and copy into pageable pages the driver then stages itself.
  let mut staging = {
    let mut guard = KERNELS.lock();
    take_staging::<R>(guard.as_mut().expect("initialised"), 0, out_len)?
  };
  stream
    .memcpy_dtoh(&d_out.slice(0..out_len), unsafe {
      staging.as_mut_slice(out_len)
    })
    .map_err(|e| driver_error("dtoh", e))?;
  // A copy into page-locked memory is asynchronous, so the buffer is only the
  // launch's output once the stream says so.
  stream
    .synchronize()
    .map_err(|e| driver_error("dtoh sync", e))?;
  let out = finish(unsafe { staging.as_slice(out_len) });
  {
    let mut guard = KERNELS.lock();
    let kernels = guard.as_mut().expect("initialised");
    return_output(kernels, 0, d_out);
    return_staging(kernels, 0, staging);
  }
  Ok(out)
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
      Reduce::None,
      |data| {
        Array2::from_shape_vec((m, process.grid_points()), data.to_vec())
          .expect("the kernel returns m * n values for a one-component family")
      },
    )
  }

  /// The fold in the kernel: `m` values back instead of `m × n`.
  ///
  /// Every step's four bytes cross PCIe in the other entry points, and that
  /// crossing is what a launch here costs — the kernel is under two per cent
  /// of the wall on a T4. A reduction is the only change that makes the
  /// crossing smaller rather than faster.
  fn euler_kernel_reduce<P: EulerCoefficients<T>>(
    &self,
    process: &P,
    first: usize,
    m: usize,
    seed: u64,
    reduce: Reduce,
  ) -> Result<Vec<T>> {
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
      reduce,
      // The first plane again: one value a path, the component a one-state
      // family reports.
      |data| data[..m.min(data.len())].to_vec(),
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
      Reduce::None,
      |data| {
        f(ndarray::ArrayView2::from_shape((m, n), data)
          .expect("the kernel returns m * n values for a one-component family"))
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
      Reduce::None,
      |data| {
        Array3::from_shape_vec((D, m, process.grid_points()), data.to_vec())
          .expect("the kernel returns components * m * n values")
      },
    )
  }

  fn batch_budget(&self) -> usize {
    self.batch_budget
  }

  /// Chunks alternate between two streams: while chunk `k` copies back
  /// through pinned memory, chunk `k + 1` is already computing.
  fn euler_kernel_batch<P: EulerCoefficients<T>>(
    &self,
    process: &P,
    m: usize,
    seed: u64,
  ) -> Result<Array2<T>> {
    let n = process.grid_points();
    let rows = crate::device::chunk_rows(self.batch_budget, n, std::mem::size_of::<T>());
    if m <= rows {
      return self.euler_kernel(process, 0, m, seed);
    }
    // The pipelined batch hashes its own Gaussian increments and knows
    // nothing of a fractional pipeline or a Markov lift, so a process with
    // either goes chunk by chunk through the single launch instead, which
    // runs the pipeline at the right row offset for each chunk. Sending it
    // through the pipeline would silently step Brownian motion in place of
    // the fractional or rough one. The history, series and table blocks live
    // in the kernel's own per-path arrays and travel as launch scalars, so
    // they ride the pipeline unchanged.
    if process.fgn_spec().is_some() || process.lift_spec().is_some() {
      let mut out = Array2::<T>::zeros((m, n));
      let mut first = 0;
      while first < m {
        let len = rows.min(m - first);
        let chunk = self.euler_kernel(process, first, len, seed)?;
        out
          .slice_mut(ndarray::s![first..first + len, ..])
          .assign(&chunk);
        first += len;
      }
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

/// The kernel launch for an explicit specification, with `finish` run over
/// the `components × m × n` values it wrote, in the caller's precision.
#[allow(clippy::too_many_arguments)]
fn device_paths<T: FloatExt, O>(
  ordinal: usize,
  spec: EulerSpec<T>,
  x0: [T; 4],
  n: usize,
  dt: T,
  first: usize,
  m: usize,
  seed: u64,
  fgn: Option<crate::euler::FgnSpec<'_, T>>,
  curves: Option<Vec<Vec<T>>>,
  jump_lambda: Option<T>,
  sizes: Option<crate::euler::JumpSizes<T>>,
  step_first: bool,
  gammas: Option<crate::euler::GammaDraws<T>>,
  lift: Option<crate::euler::LiftSpec<'_, T>>,
  series: Option<u32>,
  table: Option<crate::euler::TableSpec<T>>,
  program: Option<crate::euler::ProgramSpec<'_>>,
  reduce: Reduce,
  finish: impl FnOnce(&[T]) -> O,
) -> Result<O> {
  {
    let (curve, n_curves) = crate::euler::flatten_curves(curves, n);
    let (program_t, program_n) = crate::euler::encode_programs::<T>(program.as_ref());
    let program64: Vec<f64> = program_t
      .iter()
      .map(|v| v.to_f64().unwrap_or(0.0))
      .collect();
    let program32: Vec<f32> = program64.iter().map(|v| *v as f32).collect();
    let (lift_tables, has_lift, lift_n, lift_db, lift_fb, lift_x0) =
      crate::euler::encode_lift(lift.as_ref());
    let lift_tables64: Vec<Vec<f64>> = lift_tables
      .iter()
      .map(|t| t.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect())
      .collect();
    let (lift_db, lift_fb, lift_x0) = (
      lift_db.to_f64().unwrap_or(0.0),
      lift_fb.to_f64().unwrap_or(0.0),
      lift_x0.to_f64().unwrap_or(0.0),
    );
    let (family, params) = spec.encode();
    let hist_slot = crate::euler::history_slot(family, n);
    let series_n = crate::euler::series_terms(family, n, series);
    let series_live = crate::euler::series_live(family);
    let (table_n, table_u0) = crate::euler::table_terms(family, table);
    let arity = super::families::Family::from_code(family).expect("a declared family");
    let use_jumps = u32::from(jump_lambda.is_some());
    let lambda64 = jump_lambda.map_or(0.0, |v| v.to_f64().unwrap_or(0.0));
    let (jump_law, ja, jb, jc) = sizes.map_or((0, 0.0, 0.0, 0.0), |s| {
      let (law, a, b, c) = s.encode();
      (
        law,
        a.to_f64().unwrap_or(0.0),
        b.to_f64().unwrap_or(0.0),
        c.to_f64().unwrap_or(0.0),
      )
    });
    let step_first = u32::from(step_first);
    let (gamma_law, gs1, gc1, gp1, gs2, gc2, gp2) =
      gammas.map_or((0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), |g| {
        let (law, s1, c1, p1, s2, c2, p2) = g.encode();
        let f = |v: T| v.to_f64().unwrap_or(0.0);
        (law, f(s1), f(c1), f(p1), f(s2), f(c2), f(p2))
      });
    let (components, noises) = (arity.components() as u32, arity.noises() as u32);
    if n == 0 || m == 0 {
      return Ok(finish(&[]));
    }
    let dt = dt.to_f64().unwrap_or(0.0);
    let seed32 = (seed ^ (seed >> 32)) as u32;
    let p64: [f64; crate::euler::PARAM_SLOTS] =
      std::array::from_fn(|i| params[i].to_f64().unwrap_or(0.0));
    let streams = fgn.as_ref().map_or(1, |spec| spec.streams) as u32;
    if TypeId::of::<T>() == TypeId::of::<f64>() {
      let incs = match fgn.as_ref() {
        Some(spec) => {
          let eigs: Vec<f64> = spec
            .sqrt_eigenvalues
            .iter()
            .map(|v| v.to_f64().unwrap_or(0.0))
            .collect();
          Some(crate::noise::fgn::cuda::sampler::sample_f64_device(
            &eigs,
            spec.n,
            spec.streams * m,
            spec.offset,
            spec.hurst,
            spec.t,
            seed,
            spec.streams * first,
            ordinal,
          )?)
        }
        None => None,
      };
      let curve64: Vec<f64> = curve.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
      return run::<f64, O>(
        ordinal,
        "double",
        p64,
        std::array::from_fn(|i| x0[i].to_f64().unwrap_or(0.0)),
        dt,
        family,
        components,
        noises,
        seed32,
        first,
        n,
        m,
        incs.as_ref().map(|slice| (slice, streams)),
        &curve64,
        n_curves,
        lambda64,
        use_jumps,
        jump_law,
        ja,
        jb,
        jc,
        step_first,
        gamma_law,
        gs1,
        gc1,
        gp1,
        gs2,
        gc2,
        gp2,
        [&lift_tables64[0], &lift_tables64[1], &lift_tables64[2]],
        has_lift,
        lift_n,
        lift_db,
        lift_fb,
        lift_x0,
        hist_slot,
        series_n,
        series_live,
        table_n,
        table_u0.to_f64().unwrap_or(0.0),
        &program64,
        program_n,
        reduce,
        // The branch this is in establishes that `T` is `f64`, so the values
        // are already in the caller's precision.
        |data| finish(unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, data.len()) }),
      );
    }
    let p32: [f32; crate::euler::PARAM_SLOTS] = std::array::from_fn(|i| p64[i] as f32);
    let lift_tables32: Vec<Vec<f32>> = lift_tables64
      .iter()
      .map(|t| t.iter().map(|v| *v as f32).collect())
      .collect();
    let incs = match fgn.as_ref() {
      Some(spec) => {
        let eigs: Vec<f32> = spec
          .sqrt_eigenvalues
          .iter()
          .map(|v| v.to_f32().unwrap_or(0.0))
          .collect();
        Some(crate::noise::fgn::cuda::sampler::sample_f32_device(
          &eigs,
          spec.n,
          spec.streams * m,
          spec.offset,
          spec.hurst,
          spec.t,
          seed,
          spec.streams * first,
          ordinal,
        )?)
      }
      None => None,
    };
    let curve32: Vec<f32> = curve.iter().map(|v| v.to_f32().unwrap_or(0.0)).collect();
    assert!(
      TypeId::of::<T>() == TypeId::of::<f32>(),
      "FloatExt is implemented for f32 and f64 only"
    );
    run::<f32, O>(
      ordinal,
      "float",
      p32,
      std::array::from_fn(|i| x0[i].to_f64().unwrap_or(0.0) as f32),
      dt as f32,
      family,
      components,
      noises,
      seed32,
      first,
      n,
      m,
      incs.as_ref().map(|slice| (slice, streams)),
      &curve32,
      n_curves,
      lambda64 as f32,
      use_jumps,
      jump_law,
      ja as f32,
      jb as f32,
      jc as f32,
      step_first,
      gamma_law,
      gs1 as f32,
      gc1 as f32,
      gp1 as f32,
      gs2 as f32,
      gc2 as f32,
      gp2 as f32,
      [&lift_tables32[0], &lift_tables32[1], &lift_tables32[2]],
      has_lift,
      lift_n,
      lift_db as f32,
      lift_fb as f32,
      lift_x0 as f32,
      hist_slot,
      series_n,
      series_live,
      table_n,
      table_u0.to_f64().unwrap_or(0.0) as f32,
      &program32,
      program_n,
      reduce,
      // The assert above says `T` is `f32` here, so the values are already
      // the caller's precision.
      |data| finish(unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, data.len()) }),
    )
  }
}

/// One chunk's launch without the copy back: the caller owns the stream.
#[allow(clippy::too_many_arguments)]
fn launch_chunk<R>(
  stream: &Arc<CudaStream>,
  func: &CudaFunction,
  params: [R; crate::euler::PARAM_SLOTS],
  x0: [R; 4],
  dt: R,
  family: u32,
  components: u32,
  noises: u32,
  seed: u32,
  first: usize,
  n: usize,
  m: usize,
  increments: Option<(&CudaSlice<R>, u32)>,
  curve: &[R],
  n_curves: u32,
  jump_lambda: R,
  use_jumps: u32,
  jump_law: u32,
  jump_a: R,
  jump_b: R,
  jump_c: R,
  step_first: u32,
  gamma_law: u32,
  g1_shape: R,
  g1_scale: R,
  g1_per: R,
  g2_shape: R,
  g2_scale: R,
  g2_per: R,
  lift: [&[R]; 3],
  has_lift: u32,
  lift_n: u32,
  lift_db: R,
  lift_fb: R,
  lift_x0: R,
  hist_slot: u32,
  series_n: u32,
  series_live: u32,
  table_n: u32,
  table_u0: R,
  program: &[R],
  program_n: u32,
  d_out: &mut CudaSlice<R>,
) -> Result<()>
where
  R: DeviceRepr + ValidAsZeroBits + Copy + num_traits::Float + CachedOut,
{
  let d_params = stream
    .clone_htod(&params[..])
    .map_err(|e| driver_error("htod params", e))?;
  let sqrt_dt = dt.sqrt();
  let (steps, paths, first_path) = (n as u32, m as u32, first as u32);
  // The kernel always binds the increment pointer; an unused slot gets one
  // element rather than a null. A supplied slice was written on this device by
  // the fGN pipeline and is bound where it lies.
  let use_incs = increments.map_or(0, |(_, streams)| streams);
  let owned;
  let d_incs = match increments {
    Some((slice, _)) => slice,
    None => {
      owned = stream
        .alloc_zeros::<R>(1)
        .map_err(|e| driver_error("alloc incs", e))?;
      &owned
    }
  };
  // The kernel always binds the curve pointer; an unused slot gets one
  // element rather than a null.
  let use_curve = n_curves;
  // The three lift tables; an unused table gets one element rather than a
  // null, as the increment and curve pointers do.
  let mut d_lift = Vec::with_capacity(3);
  for table in lift {
    d_lift.push(if table.is_empty() {
      stream
        .alloc_zeros::<R>(1)
        .map_err(|e| driver_error("alloc lift", e))?
    } else {
      stream
        .clone_htod(table)
        .map_err(|e| driver_error("htod lift", e))?
    });
  }
  let d_curve = if curve.is_empty() {
    stream
      .alloc_zeros::<R>(1)
      .map_err(|e| driver_error("alloc curve", e))?
  } else {
    stream
      .clone_htod(curve)
      .map_err(|e| driver_error("htod curve", e))?
  };
  let d_program = stream
    .clone_htod(program)
    .map_err(|e| driver_error("htod program", e))?;
  unsafe {
    stream
      .launch_builder(func)
      .arg(&mut *d_out)
      .arg(&d_params)
      .arg(&family)
      .arg(&components)
      .arg(&noises)
      .arg(&x0[0])
      .arg(&x0[1])
      .arg(&x0[2])
      .arg(&x0[3])
      .arg(&dt)
      .arg(&sqrt_dt)
      .arg(&seed)
      .arg(&steps)
      .arg(&paths)
      .arg(&first_path)
      .arg(d_incs)
      .arg(&use_incs)
      .arg(&d_curve)
      .arg(&use_curve)
      .arg(&jump_lambda)
      .arg(&use_jumps)
      .arg(&jump_law)
      .arg(&jump_a)
      .arg(&jump_b)
      .arg(&jump_c)
      .arg(&step_first)
      .arg(&gamma_law)
      .arg(&g1_shape)
      .arg(&g1_scale)
      .arg(&g1_per)
      .arg(&g2_shape)
      .arg(&g2_scale)
      .arg(&g2_per)
      .arg(&d_lift[0])
      .arg(&d_lift[1])
      .arg(&d_lift[2])
      .arg(&has_lift)
      .arg(&lift_n)
      .arg(&lift_db)
      .arg(&lift_fb)
      .arg(&lift_x0)
      .arg(&hist_slot)
      .arg(&series_n)
      .arg(&series_live)
      .arg(&table_n)
      .arg(&table_u0)
      .arg(&d_program)
      .arg(&program_n)
      .launch(paths_config(paths))
      .map_err(|e| DeviceError::Launch(format!("euler_paths: {e}")))?;
  }
  Ok(())
}

/// The whole batch through the two-stream pipeline, `rows` paths per chunk.
#[allow(clippy::too_many_arguments)]
fn pipelined<R>(
  ordinal: usize,
  real: &'static str,
  params: [R; crate::euler::PARAM_SLOTS],
  x0: [R; 4],
  dt: R,
  family: u32,
  components: u32,
  noises: u32,
  seed: u32,
  n: usize,
  m: usize,
  rows: usize,
  curve: &[R],
  n_curves: u32,
  jump_lambda: R,
  use_jumps: u32,
  jump_law: u32,
  jump_a: R,
  jump_b: R,
  jump_c: R,
  step_first: u32,
  gamma_law: u32,
  g1_shape: R,
  g1_scale: R,
  g1_per: R,
  g2_shape: R,
  g2_scale: R,
  g2_per: R,
  hist_slot: u32,
  series_n: u32,
  series_live: u32,
  table_n: u32,
  table_u0: R,
  program: &[R],
  program_n: u32,
) -> Result<Vec<R>>
where
  R: DeviceRepr + ValidAsZeroBits + Copy + num_traits::Float + Send + Sync + CachedOut,
{
  let shape = Shape::new(
    super::families::Family::from_code(family).expect("a declared family"),
    use_jumps != 0 || jump_law != 0,
    gamma_law != 0,
  );
  ensure_kernels(ordinal, shape, real)?;
  let guard = KERNELS.lock();
  let kernels = guard.as_ref().expect("initialised");
  let streams = [kernels.stream.clone(), kernels.stream_b.clone()];
  let func = kernels
    .functions
    .get(&(shape, real))
    .expect("compiled for this shape")
    .clone();
  drop(guard);
  let planes = components as usize;
  let mut host = vec![R::zero(); planes * m * n];
  // Both staging buffers come from the cache: pinning pages is a syscall
  // that scales with the allocation, and at the default budget these are a
  // gigabyte each — paid on every pipelined batch before this.
  let staging = {
    let mut guard = KERNELS.lock();
    let kernels = guard.as_mut().expect("initialised");
    [
      take_staging::<R>(kernels, 0, planes * rows * n)?,
      take_staging::<R>(kernels, 1, planes * rows * n)?,
    ]
  };
  // Per slot: the device buffer kept alive until its copy has landed, and
  // the `(first, len)` rows the staging buffer holds.
  let mut in_flight: [Option<(usize, usize)>; 2] = [None, None];
  let drain =
    |slot: usize, in_flight: &mut [Option<(usize, usize)>; 2], host: &mut [R]| -> Result<()> {
      if let Some((f0, l0)) = in_flight[slot].take() {
        streams[slot]
          .synchronize()
          .map_err(|e| DeviceError::Launch(format!("sync chunk: {e}")))?;
        let src = unsafe { std::slice::from_raw_parts(staging[slot].ptr, planes * l0 * n) };
        // A chunk holds its own planes back to back; the batch holds each
        // plane whole, so the rows land one plane at a time.
        for c in 0..planes {
          let to = (c * m + f0) * n;
          host[to..to + l0 * n].copy_from_slice(&src[c * l0 * n..(c + 1) * l0 * n]);
        }
      }
      Ok(())
    };
  // The two output buffers the pipeline alternates between, sized for the
  // largest chunk and kept across calls: allocating them per chunk is what
  // made a Metal batch three times slower than its kernel.
  let mut buffers = {
    let mut guard = KERNELS.lock();
    let kernels = guard.as_mut().expect("initialised");
    let need = planes * rows * n;
    [
      take_output::<R>(kernels, &streams[0], 0, need)?,
      take_output::<R>(kernels, &streams[1], 1, need)?,
    ]
  };
  let mut first = 0;
  let mut k = 0;
  while first < m {
    let len = rows.min(m - first);
    let slot = k % 2;
    drain(slot, &mut in_flight, &mut host)?;
    launch_chunk(
      &streams[slot],
      &func,
      params,
      x0,
      dt,
      family,
      components,
      noises,
      seed,
      first,
      n,
      len,
      // The pipelined batch is Gaussian only: `euler_kernel_batch` routes a
      // fractional process around it, chunk by chunk through `euler_kernel`,
      // so no increment buffer ever meets this launch.
      None,
      curve,
      n_curves,
      jump_lambda,
      use_jumps,
      jump_law,
      jump_a,
      jump_b,
      jump_c,
      step_first,
      gamma_law,
      g1_shape,
      g1_scale,
      g1_per,
      g2_shape,
      g2_scale,
      g2_per,
      [&[], &[], &[]],
      0,
      0,
      R::zero(),
      R::zero(),
      R::zero(),
      hist_slot,
      series_n,
      series_live,
      table_n,
      table_u0,
      program,
      program_n,
      &mut buffers[slot],
    )?;
    let dst = unsafe { std::slice::from_raw_parts_mut(staging[slot].ptr, planes * len * n) };
    streams[slot]
      .memcpy_dtoh(&buffers[slot].slice(0..planes * len * n), dst)
      .map_err(|e| driver_error("dtoh chunk", e))?;
    in_flight[slot] = Some((first, len));
    first += len;
    k += 1;
  }
  drain(0, &mut in_flight, &mut host)?;
  drain(1, &mut in_flight, &mut host)?;
  {
    let mut guard = KERNELS.lock();
    let kernels = guard.as_mut().expect("initialised");
    let [b0, b1] = buffers;
    return_output(kernels, 0, b0);
    return_output(kernels, 1, b1);
    let [s0, s1] = staging;
    return_staging(kernels, 0, s0);
    return_staging(kernels, 1, s1);
  }
  Ok(host)
}

/// The pipelined batch for an explicit specification, in the precision of `T`.
#[allow(clippy::too_many_arguments)]
fn pipelined_paths<T: FloatExt>(
  ordinal: usize,
  spec: EulerSpec<T>,
  x0: [T; 4],
  n: usize,
  dt: T,
  m: usize,
  rows: usize,
  seed: u64,
  curves: Option<Vec<Vec<T>>>,
  jump_lambda: Option<T>,
  sizes: Option<crate::euler::JumpSizes<T>>,
  step_first: bool,
  gammas: Option<crate::euler::GammaDraws<T>>,
  lift: Option<crate::euler::LiftSpec<'_, T>>,
  series: Option<u32>,
  table: Option<crate::euler::TableSpec<T>>,
  program: Option<crate::euler::ProgramSpec<'_>>,
) -> Result<Array3<T>> {
  let (curve, n_curves) = crate::euler::flatten_curves(curves, n);
  let (program_t, program_n) = crate::euler::encode_programs::<T>(program.as_ref());
  let program64: Vec<f64> = program_t
    .iter()
    .map(|v| v.to_f64().unwrap_or(0.0))
    .collect();
  let program32: Vec<f32> = program64.iter().map(|v| *v as f32).collect();
  debug_assert!(lift.is_none(), "the pipelined batch carries no lift");
  let (family, params) = spec.encode();
  let hist_slot = crate::euler::history_slot(family, n);
  let series_n = crate::euler::series_terms(family, n, series);
  let series_live = crate::euler::series_live(family);
  let (table_n, table_u0) = crate::euler::table_terms(family, table);
  let arity = super::families::Family::from_code(family).expect("a declared family");
  let use_jumps = u32::from(jump_lambda.is_some());
  let lambda64 = jump_lambda.map_or(0.0, |v| v.to_f64().unwrap_or(0.0));
  let (jump_law, ja, jb, jc) = sizes.map_or((0, 0.0, 0.0, 0.0), |s| {
    let (law, a, b, c) = s.encode();
    (
      law,
      a.to_f64().unwrap_or(0.0),
      b.to_f64().unwrap_or(0.0),
      c.to_f64().unwrap_or(0.0),
    )
  });
  let step_first = u32::from(step_first);
  let (gamma_law, gs1, gc1, gp1, gs2, gc2, gp2) =
    gammas.map_or((0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), |g| {
      let (law, s1, c1, p1, s2, c2, p2) = g.encode();
      let f = |v: T| v.to_f64().unwrap_or(0.0);
      (law, f(s1), f(c1), f(p1), f(s2), f(c2), f(p2))
    });
  let (components, noises) = (arity.components() as u32, arity.noises() as u32);
  let planes = components as usize;
  let dt = dt.to_f64().unwrap_or(0.0);
  let seed32 = (seed ^ (seed >> 32)) as u32;
  let p64: [f64; crate::euler::PARAM_SLOTS] =
    std::array::from_fn(|i| params[i].to_f64().unwrap_or(0.0));
  if TypeId::of::<T>() == TypeId::of::<f64>() {
    let curve64: Vec<f64> = curve.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
    let data = pipelined::<f64>(
      ordinal,
      "double",
      p64,
      std::array::from_fn(|i| x0[i].to_f64().unwrap_or(0.0)),
      dt,
      family,
      components,
      noises,
      seed32,
      n,
      m,
      rows,
      &curve64,
      n_curves,
      lambda64,
      use_jumps,
      jump_law,
      ja,
      jb,
      jc,
      step_first,
      gamma_law,
      gs1,
      gc1,
      gp1,
      gs2,
      gc2,
      gp2,
      hist_slot,
      series_n,
      series_live,
      table_n,
      table_u0.to_f64().unwrap_or(0.0),
      &program64,
      program_n,
    )?;
    let out = Array3::<f64>::from_shape_vec((planes, m, n), data)
      .expect("the kernel returns components * m * n values");
    return Ok(unsafe { std::mem::transmute::<Array3<f64>, Array3<T>>(out) });
  }
  assert!(
    TypeId::of::<T>() == TypeId::of::<f32>(),
    "FloatExt is implemented for f32 and f64 only"
  );
  let p32: [f32; crate::euler::PARAM_SLOTS] = std::array::from_fn(|i| p64[i] as f32);
  let curve32: Vec<f32> = curve.iter().map(|v| v.to_f32().unwrap_or(0.0)).collect();
  let data = pipelined::<f32>(
    ordinal,
    "float",
    p32,
    std::array::from_fn(|i| x0[i].to_f64().unwrap_or(0.0) as f32),
    dt as f32,
    family,
    components,
    noises,
    seed32,
    n,
    m,
    rows,
    &curve32,
    n_curves,
    lambda64 as f32,
    use_jumps,
    jump_law,
    ja as f32,
    jb as f32,
    jc as f32,
    step_first,
    gamma_law,
    gs1 as f32,
    gc1 as f32,
    gp1 as f32,
    gs2 as f32,
    gc2 as f32,
    gp2 as f32,
    hist_slot,
    series_n,
    series_live,
    table_n,
    table_u0.to_f64().unwrap_or(0.0) as f32,
    &program32,
    program_n,
  )?;
  let out = Array3::<f32>::from_shape_vec((planes, m, n), data)
    .expect("the kernel returns components * m * n values");
  Ok(unsafe { std::mem::transmute::<Array3<f32>, Array3<T>>(out) })
}
