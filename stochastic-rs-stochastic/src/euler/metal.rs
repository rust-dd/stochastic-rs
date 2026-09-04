//! Native Metal device path of the Euler engine (macOS, `metal` feature):
//! hand-written MSL, one thread per path, the whole recursion in the kernel,
//! normals from the same counter hash of `(path, step, seed)` as the CubeCL
//! and CUDA kernels. `f32` only — Apple GPUs have no double precision — and
//! widened on the way back.

use metal::*;
use ndarray::Array2;
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
    float x0;
    float dt;
    float sqrt_dt;
    uint seed;
    uint steps;
    uint paths;
    uint first_path;
    uint increments;
};

kernel void euler_paths(
    device float* out [[buffer(0)]],
    device const float* params [[buffer(1)]],
    constant EulerArgs& args [[buffer(2)]],
    device const float* incs [[buffer(3)]],
    uint path [[thread_position_in_grid]])
{
    const uint family = args.family;
    const float x0 = args.x0;
    const float dt = args.dt;
    const float sqrt_dt = args.sqrt_dt;
    const uint seed = args.seed;
    const uint steps = args.steps;
    const uint paths = args.paths;
    const uint first_path = args.first_path;
    const uint increments = args.increments;
"#;

fn msl_source() -> String {
  let lang = super::kernel::Language {
    real: "float",
    sqrt: "sqrt",
    log: "log",
    cos: "cos",
    exp: "exp",
    pow: "pow",
    abs: "abs",
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
  x0: f32,
  dt: f32,
  sqrt_dt: f32,
  seed: u32,
  steps: u32,
  paths: u32,
  first_path: u32,
  /// Non-zero when the launch reads its noise from the increment buffer
  /// rather than hashing it.
  increments: u32,
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
  params: [f32; 4],
  args: EulerArgs,
  increments: Increments<'_>,
) -> Result<Vec<f32>> {
  ensure_context(ordinal)?;
  let guard = CONTEXT.lock();
  let ctx = guard.as_ref().expect("initialised");
  let shared = MTLResourceOptions::StorageModeShared;
  let total = args.paths as usize * args.steps as usize;
  let out_buf = ctx.device.new_buffer((total * 4) as u64, shared);
  let params_buf = ctx
    .device
    .new_buffer_with_data(params.as_ptr() as *const _, 16, shared);
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
  let cmd = ctx.queue.new_command_buffer();
  {
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.pipeline);
    enc.set_buffer(0, Some(&out_buf), 0);
    enc.set_buffer(1, Some(&params_buf), 0);
    enc.set_buffer(3, Some(incs_buf), 0);
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
    device_paths(
      self.ordinal,
      process.euler_spec(),
      process.initial_value(),
      process.grid_points(),
      process.horizon(),
      first,
      m,
      seed,
      match fractional.as_ref() {
        Some(buf) => Increments::Device(buf),
        None => Increments::Hashed,
      },
    )
  }

  fn batch_budget(&self) -> usize {
    self.batch_budget
  }
}

/// The kernel launch for an explicit specification.
#[allow(clippy::too_many_arguments)]
fn device_paths(
  ordinal: usize,
  spec: EulerSpec<f32>,
  x0: f32,
  n: usize,
  t: f32,
  first: usize,
  m: usize,
  seed: u64,
  increments: Increments<'_>,
) -> Result<Array2<f32>> {
  {
    if n == 0 || m == 0 {
      return Ok(Array2::<f32>::zeros((m, n)));
    }
    let (family, params) = spec.encode();
    let dt = (t as f64 / (n.max(2) - 1) as f64) as f32;
    let params32: [f32; 4] = params;
    let args = EulerArgs {
      family,
      x0,
      dt,
      sqrt_dt: dt.sqrt(),
      seed: (seed ^ (seed >> 32)) as u32,
      steps: n as u32,
      paths: m as u32,
      first_path: first as u32,
      increments: u32::from(!matches!(increments, Increments::Hashed)),
    };
    let data = run(ordinal, params32, args, increments)?;
    Ok(Array2::from_shape_vec((m, n), data).expect("the kernel returns m * n values"))
  }
}
