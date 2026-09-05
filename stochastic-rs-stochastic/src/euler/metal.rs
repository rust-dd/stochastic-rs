//! Native Metal device path of the Euler engine (macOS, `metal` feature):
//! hand-written MSL, one thread per path, the whole recursion in the kernel,
//! normals from the same counter hash of `(path, step, seed)` as the CubeCL
//! and CUDA kernels. `f32` only — Apple GPUs have no double precision — and
//! widened on the way back.

use metal::*;
use ndarray::Array2;
use ndarray::Array3;
use parking_lot::Mutex;

use super::EulerCoefficients;
use super::EulerKernel;
use super::EulerSpec;
use crate::device::DeviceError;
use crate::device::DeviceInfo;
use crate::device::Metal;

type Result<T> = std::result::Result<T, DeviceError>;

/// The MSL header around the kernel body the CUDA back-end renders too
/// ([`super::kernel`]): the argument struct, the kernel signature and the
/// bindings of its fields to the names the body uses.
const MSL_HEADER: &str = r#"#include <metal_stdlib>
using namespace metal;

struct EulerArgs {
    uint family;
    uint components;
    uint noises;
    uint seed;
    uint steps;
    uint paths;
    uint first_path;
    uint increments;
    uint has_curve;
    uint has_jumps;
    uint jump_law;
    uint step_first;
    uint gamma_law;
    float dt;
    float sqrt_dt;
    float jump_lambda;
    float jump_a;
    float jump_b;
    float jump_c;
    float g1_shape;
    float g1_scale;
    float g2_shape;
    float g2_scale;
    float x0[4];
};

kernel void euler_paths(
    device float* out [[buffer(0)]],
    device const float* params [[buffer(1)]],
    constant EulerArgs& args [[buffer(2)]],
    device const float* incs [[buffer(3)]],
    device const float* curve [[buffer(4)]],
    uint path [[thread_position_in_grid]])
{
    const uint family = args.family;
    const uint components = args.components;
    const uint noises = args.noises;
    const float dt = args.dt;
    const float sqrt_dt = args.sqrt_dt;
    const uint seed = args.seed;
    const uint steps = args.steps;
    const uint paths = args.paths;
    const uint first_path = args.first_path;
    const uint increments = args.increments;
    const uint has_curve = args.has_curve;
    const uint has_jumps = args.has_jumps;
    const float jump_lambda = args.jump_lambda;
    const uint jump_law = args.jump_law;
    const uint step_first = args.step_first;
    const uint gamma_law = args.gamma_law;
    const float g1_shape = args.g1_shape;
    const float g1_scale = args.g1_scale;
    const float g2_shape = args.g2_shape;
    const float g2_scale = args.g2_scale;
    const float jump_a = args.jump_a;
    const float jump_b = args.jump_b;
    const float jump_c = args.jump_c;
    const float x0[4] = { args.x0[0], args.x0[1], args.x0[2], args.x0[3] };
"#;

fn msl_source() -> String {
  let lang = super::kernel::Language {
    real: "float",
    sqrt: "sqrt",
    log: "log",
    cos: "cos",
    sin: "sin",
    exp: "exp",
    pow: "pow",
    abs: "abs",
    tanh: "tanh",
    index: "uint",
  };
  let prelude = super::kernel::prelude(&lang);
  let body = super::kernel::render(&lang);
  format!("{prelude}{MSL_HEADER}{body}}}\n")
}

#[repr(C)]
#[derive(Clone, Copy)]
struct EulerArgs {
  family: u32,
  /// State components the family steps, and planes the launch writes.
  components: u32,
  /// Independent noise components the step draws.
  noises: u32,
  seed: u32,
  steps: u32,
  paths: u32,
  first_path: u32,
  /// Non-zero when the launch reads its first noise component from the
  /// increment buffer rather than hashing it.
  increments: u32,
  /// Non-zero when the launch binds a time-varying coefficient.
  has_curve: u32,
  /// Non-zero when the launch draws a jump count per step.
  has_jumps: u32,
  /// Which size law the jumps carry: none, normal, or double-exponential.
  jump_law: u32,
  /// Non-zero when the first point written is a step rather than the
  /// initial state.
  step_first: u32,
  /// How many Gamma draws the step takes: none, one, or two.
  gamma_law: u32,
  dt: f32,
  sqrt_dt: f32,
  jump_lambda: f32,
  jump_a: f32,
  jump_b: f32,
  jump_c: f32,
  g1_shape: f32,
  g1_scale: f32,
  g2_shape: f32,
  g2_scale: f32,
  x0: [f32; 4],
}

struct Context {
  ordinal: usize,
  device: Device,
  queue: CommandQueue,
  pipeline: ComputePipelineState,
}

/// SAFETY: every device operation is serialised through the one queue.
unsafe impl Send for Context {}

static CONTEXT: Mutex<Option<Context>> = Mutex::new(None);

/// The Metal device at `ordinal`: index `0` is the system default, `n > 0` the
/// n-th entry of `Device::all()`.
pub(crate) fn metal_device(ordinal: usize) -> Result<Device> {
  if ordinal == 0 {
    return Device::system_default()
      .ok_or_else(|| DeviceError::Unavailable("no Metal device".to_string()));
  }
  let mut all = Device::all();
  let count = all.len();
  if ordinal < count {
    Ok(all.swap_remove(ordinal))
  } else {
    Err(DeviceError::Unavailable(format!(
      "no Metal device at index {ordinal}; this machine has {count}"
    )))
  }
}

/// The Metal device at `ordinal`, or why it cannot be used.
pub(crate) fn probe(ordinal: usize) -> Result<DeviceInfo> {
  let device = metal_device(ordinal)?;
  Ok(DeviceInfo::new(
    "Metal",
    device.name().to_string(),
    &["f32"],
    Some(ordinal),
  ))
}

fn ensure_context(ordinal: usize) -> Result<()> {
  let mut guard = CONTEXT.lock();
  if guard.as_ref().is_some_and(|c| c.ordinal == ordinal) {
    return Ok(());
  }
  *guard = None;
  let device = metal_device(ordinal)?;
  let queue = device.new_command_queue();
  let library = device
    .new_library_with_source(&msl_source(), &CompileOptions::new())
    .map_err(|e| DeviceError::Compile(format!("MSL compile: {e}")))?;
  let function = library
    .get_function("euler_paths", None)
    .map_err(|e| DeviceError::Launch(format!("get euler_paths: {e}")))?;
  let pipeline = device
    .new_compute_pipeline_state_with_function(&function)
    .map_err(|e| DeviceError::Launch(format!("euler_paths PSO: {e}")))?;
  *guard = Some(Context {
    ordinal,
    device,
    queue,
    pipeline,
  });
  Ok(())
}

/// Where a launch's noise increments come from. There is no host variant:
/// increments are either hashed in the kernel or produced on this device, so
/// they never travel through host memory.
pub(crate) enum Increments<'a> {
  /// The kernel hashes its own Gaussian increments.
  Hashed,
  /// A buffer already written on this device — the fGN pipeline's own
  /// output, which never leaves the GPU.
  Device(&'a Buffer),
}

fn run(
  ordinal: usize,
  params: [f32; crate::euler::PARAM_SLOTS],
  args: EulerArgs,
  increments: Increments<'_>,
  curve: &[f32],
) -> Result<Vec<f32>> {
  ensure_context(ordinal)?;
  let guard = CONTEXT.lock();
  let ctx = guard.as_ref().expect("initialised");
  let shared = MTLResourceOptions::StorageModeShared;
  let total = args.components as usize * args.paths as usize * args.steps as usize;
  let out_buf = ctx.device.new_buffer((total * 4) as u64, shared);
  let params_buf = ctx.device.new_buffer_with_data(
    params.as_ptr() as *const _,
    (crate::euler::PARAM_SLOTS * 4) as u64,
    shared,
  );
  // Metal requires every declared buffer to be bound, so an unused increment
  // slot still gets one float. A supplied buffer is bound as it stands: it was
  // written on this device and never left it.
  let owned;
  let incs_buf = match increments {
    Increments::Device(buf) => buf,
    Increments::Hashed => {
      owned = ctx.device.new_buffer(4, shared);
      &owned
    }
  };
  // As with the increment slot, an unused curve still gets a bound buffer.
  let curve_buf = if curve.is_empty() {
    ctx.device.new_buffer(4, shared)
  } else {
    ctx.device.new_buffer_with_data(
      curve.as_ptr() as *const _,
      std::mem::size_of_val(curve) as u64,
      shared,
    )
  };
  let cmd = ctx.queue.new_command_buffer();
  {
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.pipeline);
    enc.set_buffer(0, Some(&out_buf), 0);
    enc.set_buffer(1, Some(&params_buf), 0);
    enc.set_buffer(3, Some(incs_buf), 0);
    enc.set_buffer(4, Some(&curve_buf), 0);
    enc.set_bytes(
      2,
      std::mem::size_of::<EulerArgs>() as u64,
      &args as *const EulerArgs as *const _,
    );
    let width = ctx.pipeline.max_total_threads_per_threadgroup().min(256);
    enc.dispatch_threads(
      MTLSize::new(args.paths as u64, 1, 1),
      MTLSize::new(width, 1, 1),
    );
    enc.end_encoding();
  }
  cmd.commit();
  cmd.wait_until_completed();
  let ptr = out_buf.contents() as *const f32;
  Ok(unsafe { std::slice::from_raw_parts(ptr, total) }.to_vec())
}

impl EulerKernel<f32> for Metal {
  fn euler_kernel<P: EulerCoefficients<f32>>(
    &self,
    process: &P,
    first: usize,
    m: usize,
    seed: u64,
  ) -> Result<Array2<f32>> {
    // A fractional process has its increments produced on this same device and
    // read from the buffer they were written to: the two kernels meet in GPU
    // memory rather than through the host.
    let fractional = match process.fgn_spec() {
      Some(spec) => {
        let eigs: Vec<f32> = spec.sqrt_eigenvalues.to_vec();
        let (buf, _) = crate::noise::fgn::metal::sample_f32_buffer(
          &eigs,
          spec.n,
          m,
          spec.offset,
          spec.hurst,
          spec.t,
          seed as u32,
          self.ordinal,
        )?;
        Some(buf)
      }
      None => None,
    };
    let planes = device_paths(
      self.ordinal,
      process.euler_spec(),
      process.initial_state(),
      process.grid_points(),
      process.time_step(),
      first,
      m,
      seed,
      match fractional.as_ref() {
        Some(buf) => Increments::Device(buf),
        None => Increments::Hashed,
      },
      process.curve().as_deref().unwrap_or(&[]),
      process.jump_intensity(),
      process.jump_sizes(),
      process.step_first(),
      process.gamma_draws(),
    )?;
    Ok(planes.index_axis_move(ndarray::Axis(0), 0))
  }

  /// A system's launch: the same kernel, its state slots filled from the
  /// process's own initial state and every component's plane returned.
  fn euler_system_kernel<const D: usize, P: super::EulerSystem<f32, D>>(
    &self,
    process: &P,
    first: usize,
    m: usize,
    seed: u64,
  ) -> Result<Array3<f32>> {
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
      Increments::Hashed,
      process.curve().as_deref().unwrap_or(&[]),
      process.jump_intensity(),
      process.jump_sizes(),
      process.step_first(),
      process.gamma_draws(),
    )
  }

  fn batch_budget(&self) -> usize {
    self.batch_budget
  }
}

/// The kernel launch for an explicit specification, as a
/// `components × m × n` array. A one-component family fills a single plane,
/// which is what [`EulerKernel::euler_kernel`] hands back as its matrix.
#[allow(clippy::too_many_arguments)]
fn device_paths(
  ordinal: usize,
  spec: EulerSpec<f32>,
  x0: [f32; 4],
  n: usize,
  dt: f32,
  first: usize,
  m: usize,
  seed: u64,
  increments: Increments<'_>,
  curve: &[f32],
  jump_lambda: Option<f32>,
  sizes: Option<crate::euler::JumpSizes<f32>>,
  step_first: bool,
  gammas: Option<crate::euler::GammaDraws<f32>>,
) -> Result<Array3<f32>> {
  let (family, params) = spec.encode();
  let arity = super::families::Family::from_code(family).expect("a declared family");
  let (components, noises) = (arity.components(), arity.noises());
  if n == 0 || m == 0 {
    return Ok(Array3::<f32>::zeros((components, m, n)));
  }
  let (law, jump_a, jump_b, jump_c) = sizes.map_or((0, 0.0, 0.0, 0.0), |s| s.encode());
  let (gamma_law, g1_shape, g1_scale, g2_shape, g2_scale) =
    gammas.map_or((0, 0.0, 0.0, 0.0, 0.0), |g| g.encode());
  let args = EulerArgs {
    family,
    components: components as u32,
    noises: noises as u32,
    seed: (seed ^ (seed >> 32)) as u32,
    steps: n as u32,
    paths: m as u32,
    first_path: first as u32,
    increments: u32::from(!matches!(increments, Increments::Hashed)),
    has_curve: u32::from(!curve.is_empty()),
    has_jumps: u32::from(jump_lambda.is_some()),
    jump_law: law,
    step_first: u32::from(step_first),
    gamma_law,
    dt,
    sqrt_dt: dt.sqrt(),
    jump_lambda: jump_lambda.unwrap_or(0.0),
    jump_a,
    jump_b,
    jump_c,
    g1_shape,
    g1_scale,
    g2_shape,
    g2_scale,
    x0,
  };
  let data = run(ordinal, params, args, increments, curve)?;
  Ok(
    Array3::from_shape_vec((components, m, n), data)
      .expect("the kernel returns components * m * n values"),
  )
}
