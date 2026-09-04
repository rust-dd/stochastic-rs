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
use crate::device::MetalNative;

type Result<T> = std::result::Result<T, DeviceError>;

const MSL_SOURCE: &str = r#"
#include <metal_stdlib>
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
};

kernel void euler_paths(
    device float* out [[buffer(0)]],
    device const float* params [[buffer(1)]],
    constant EulerArgs& args [[buffer(2)]],
    uint path [[thread_position_in_grid]])
{
    if (path >= args.paths) return;
    uint base = path * args.steps;
    float x = args.x0;
    float reported = args.x0;
    if (args.family == 2u && args.x0 < 0.0f) reported = 0.0f;
    out[base] = reported;
    for (uint i = 1u; i < args.steps; i++) {
        uint g = (args.first_path + path) * args.steps + i;
        uint a = (g * 2u) ^ (args.seed * 2654435761u);
        a ^= a >> 16; a *= 2246822519u; a ^= a >> 13; a *= 3266489917u; a ^= a >> 16;
        uint b = (g * 2u + 1u) ^ (args.seed * 668265263u);
        b ^= b >> 16; b *= 2246822519u; b ^= b >> 13; b *= 3266489917u; b ^= b >> 16;
        float u1 = float(a) * 2.3283064e-10f * 0.999998f + 1.0e-6f;
        float u2 = float(b) * 2.3283064e-10f;
        float z = sqrt(-2.0f * log(u1)) * cos(6.2831853071795864f * u2);
        if (args.family == 0u) {
            x = x + params[0] * x * args.dt + params[1] * x * args.sqrt_dt * z;
        } else if (args.family == 1u) {
            x = x + params[0] * (params[1] - x) * args.dt + params[2] * args.sqrt_dt * z;
        } else {
            float positive = x > 0.0f ? x : 0.0f;
            x = x + params[0] * (params[1] - positive) * args.dt + params[2] * sqrt(positive) * args.sqrt_dt * z;
        }
        reported = x;
        if (args.family == 2u && x < 0.0f) reported = 0.0f;
        out[base + i] = reported;
    }
}
"#;

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
    "MetalNative",
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
    .new_library_with_source(MSL_SOURCE, &CompileOptions::new())
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

fn run(ordinal: usize, params: [f32; 4], args: EulerArgs) -> Result<Vec<f32>> {
  ensure_context(ordinal)?;
  let guard = CONTEXT.lock();
  let ctx = guard.as_ref().expect("initialised");
  let shared = MTLResourceOptions::StorageModeShared;
  let total = args.paths as usize * args.steps as usize;
  let out_buf = ctx.device.new_buffer((total * 4) as u64, shared);
  let params_buf = ctx
    .device
    .new_buffer_with_data(params.as_ptr() as *const _, 16, shared);
  let cmd = ctx.queue.new_command_buffer();
  {
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.pipeline);
    enc.set_buffer(0, Some(&out_buf), 0);
    enc.set_buffer(1, Some(&params_buf), 0);
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

impl EulerKernel<f32> for MetalNative {
  fn euler_kernel<P: EulerCoefficients<f32>>(
    &self,
    process: &P,
    first: usize,
    m: usize,
    seed: u64,
  ) -> Result<Array2<f32>> {
    device_paths(
      self.ordinal,
      process.euler_spec(),
      process.initial_value(),
      process.grid_points(),
      process.horizon(),
      first,
      m,
      seed,
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
    };
    let data = run(ordinal, params32, args)?;
    Ok(Array2::from_shape_vec((m, n), data).expect("the kernel returns m * n values"))
  }
}
