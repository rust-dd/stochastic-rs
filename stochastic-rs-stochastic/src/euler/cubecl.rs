//! CubeCL device path of the Euler engine: one thread per path, the whole
//! recursion in the kernel, normals from a counter hash of
//! `(path, step, seed)` pushed through Box–Muller — the same generator as
//! the fGN device kernels. `f32` on the device (the portable GPU float),
//! widened on the way back.

use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use ndarray::Array2;
use parking_lot::Mutex;

use super::EulerCoefficients;
use super::EulerKernel;
use super::EulerSpec;
use super::families::cube_report;
use super::families::cube_step;
use crate::device::DeviceError;
use crate::device::DeviceInfo;

type DeviceResult<T> = std::result::Result<T, DeviceError>;

const WG_SIZE: u32 = 256;

#[cube(launch)]
fn euler_paths_kernel(
  out: &mut Array<f32>,
  params: &Array<f32>,
  family: u32,
  x0: f32,
  dt: f32,
  sqrt_dt: f32,
  seed: u32,
  steps: u32,
  paths: u32,
  first_path: u32,
) {
  let path = ABSOLUTE_POS as u32;
  if path < paths {
    let base = (path * steps) as usize;
    let mut x = x0;
    out[base] = report(family, x0);
    for i in 1..steps {
      // Two decorrelated uniforms via integer hashing (Murmur3-style finalizer).
      let g = (first_path + path) * steps + i;
      let mut a = (g * 2u32) ^ (seed * 2654435761u32);
      a ^= a >> 16;
      a *= 2246822519u32;
      a ^= a >> 13;
      a *= 3266489917u32;
      a ^= a >> 16;
      let mut b = (g * 2u32 + 1u32) ^ (seed * 668265263u32);
      b ^= b >> 16;
      b *= 2246822519u32;
      b ^= b >> 13;
      b *= 3266489917u32;
      b ^= b >> 16;
      let inv = 2.3283064e-10f32;
      let u1 = f32::cast_from(a) * inv * 0.999998f32 + 1.0e-6f32;
      let u2 = f32::cast_from(b) * inv;
      let z = Sqrt::sqrt(-2.0f32 * Log::ln(u1)) * Cos::cos(core::f32::consts::TAU * u2);
      x = step(family, x, params, dt, sqrt_dt, z);
      out[base + i as usize] = report(family, x);
    }
  }
}

/// Dispatches to the family's generated step. The formulas live in the
/// declarations in [`super::families`]; what stands here is the parameter
/// order each family reads from the buffer, which the compiler checks by
/// arity.
#[cube]
fn step(family: u32, x: f32, params: &Array<f32>, dt: f32, sqrt_dt: f32, z: f32) -> f32 {
  let mut stepped = x;
  if family == 0u32 {
    stepped = cube_step::GeometricBrownian(x, params[0], params[1], dt, sqrt_dt, z);
  }
  if family == 1u32 {
    stepped = cube_step::OrnsteinUhlenbeck(x, params[0], params[1], params[2], dt, sqrt_dt, z);
  }
  if family == 2u32 {
    stepped = cube_step::SquareRoot(x, params[0], params[1], params[2], dt, sqrt_dt, z);
  }
  stepped
}

/// Dispatches to the family's generated report.
#[cube]
fn report(family: u32, x: f32) -> f32 {
  let mut reported = x;
  if family == 0u32 {
    reported = cube_report::GeometricBrownian(x);
  }
  if family == 1u32 {
    reported = cube_report::OrnsteinUhlenbeck(x);
  }
  if family == 2u32 {
    reported = cube_report::SquareRoot(x);
  }
  reported
}

/// A CubeCL runtime this crate can open, with its own cached compute client.
/// One implementor per `cubecl-*` feature, so a build with both reaches both
/// devices; the kernels themselves are runtime-agnostic.
pub(crate) trait CubeclRuntime: 'static {
  /// The CubeCL runtime this opens.
  type Rt: cubecl::Runtime;

  /// What [`DeviceInfo::backend`] reports.
  const BACKEND: &'static str;

  /// The runtime's device at `ordinal`.
  fn device(ordinal: usize) -> <Self::Rt as cubecl::Runtime>::Device;

  /// The cached client for `ordinal`, opened on first use.
  fn client(ordinal: usize) -> DeviceResult<ComputeClient<Self::Rt>>;
}

/// The client cache of one runtime: CubeCL clients are cheap to clone and
/// expensive to open, and switching ordinal re-opens.
pub(crate) struct Context<Rt: cubecl::Runtime> {
  ordinal: usize,
  client: ComputeClient<Rt>,
}

// SAFETY: the client is only ever handed out as a clone under the mutex, and
// CubeCL's own client is internally synchronised.
unsafe impl<Rt: cubecl::Runtime> Send for Context<Rt> {}

/// The cached client for `ordinal`, re-opening when the ordinal changes.
/// CubeCL panics rather than erroring when no device exists, so the opening
/// is caught and reported as a [`DeviceError`].
pub(crate) fn open<Rt: cubecl::Runtime>(
  slot: &Mutex<Option<Context<Rt>>>,
  ordinal: usize,
  device: fn(usize) -> Rt::Device,
) -> DeviceResult<ComputeClient<Rt>> {
  let mut guard = slot.lock();
  if !guard.as_ref().is_some_and(|c| c.ordinal == ordinal) {
    *guard = None;
    // The device is built inside the closure: a `Runtime::Device` reference
    // is not `RefUnwindSafe`, a `usize` and a fn pointer are.
    match std::panic::catch_unwind(|| Rt::client(&device(ordinal))) {
      Ok(client) => *guard = Some(Context { ordinal, client }),
      Err(payload) => return Err(DeviceError::Unavailable(crate::device::panic_text(payload))),
    }
  }
  Ok(guard.as_ref().expect("initialised").client.clone())
}

/// The CUDA runtime of CubeCL, distinct from the hand-written [`Cuda`
/// ](crate::device::Cuda) backend that reaches the same hardware through
/// cudarc.
#[cfg(feature = "cubecl-cuda")]
pub(crate) mod cuda_rt {
  use super::*;

  /// The tag type; the handle is [`CubeclCuda`](crate::device::CubeclCuda).
  pub(crate) struct Rt;

  static CONTEXT: Mutex<Option<Context<cubecl_cuda::CudaRuntime>>> = Mutex::new(None);

  impl CubeclRuntime for Rt {
    type Rt = cubecl_cuda::CudaRuntime;

    const BACKEND: &'static str = "CubeclCuda";

    fn device(ordinal: usize) -> cubecl_cuda::CudaDevice {
      cubecl_cuda::CudaDevice { index: ordinal }
    }

    fn client(ordinal: usize) -> DeviceResult<ComputeClient<Self::Rt>> {
      open(&CONTEXT, ordinal, Self::device)
    }
  }
}

/// The wgpu runtime of CubeCL: Metal on macOS, Vulkan on Linux, WebGPU on the
/// web. `ordinal` `0` is the default adapter, `n > 0` the n-th discrete GPU.
#[cfg(feature = "cubecl-wgpu")]
pub(crate) mod wgpu_rt {
  use super::*;

  /// The tag type; the handle is [`CubeclWgpu`](crate::device::CubeclWgpu).
  pub(crate) struct Rt;

  static CONTEXT: Mutex<Option<Context<cubecl_wgpu::WgpuRuntime>>> = Mutex::new(None);

  impl CubeclRuntime for Rt {
    type Rt = cubecl_wgpu::WgpuRuntime;

    const BACKEND: &'static str = "CubeclWgpu";

    fn device(ordinal: usize) -> cubecl_wgpu::WgpuDevice {
      if ordinal == 0 {
        cubecl_wgpu::WgpuDevice::default()
      } else {
        cubecl_wgpu::WgpuDevice::DiscreteGpu(ordinal)
      }
    }

    fn client(ordinal: usize) -> DeviceResult<ComputeClient<Self::Rt>> {
      open(&CONTEXT, ordinal, Self::device)
    }
  }
}

/// The runtime's device at `ordinal`, or why it cannot be used.
pub(crate) fn probe<C: CubeclRuntime>(ordinal: usize) -> DeviceResult<DeviceInfo> {
  let cl = C::client(ordinal)?;
  Ok(DeviceInfo::new(
    C::BACKEND,
    <C::Rt as cubecl::Runtime>::name(&cl).to_string(),
    &["f32"],
    Some(ordinal),
  ))
}

/// Splits a 1-D cube count into a 2-D grid so no dimension exceeds WebGPU's
/// 65535 per-dimension limit.
fn count_2d(cubes: u32) -> CubeCount {
  if cubes <= 65535 {
    CubeCount::Static(cubes.max(1), 1, 1)
  } else {
    let mut x = cubes;
    let mut y = 1u32;
    while x > 32768 {
      x = x.div_ceil(2);
      y *= 2;
    }
    CubeCount::Static(x, y, 1)
  }
}

#[cfg(any(feature = "cubecl-cuda", feature = "cubecl-wgpu"))]
impl EulerKernel<f32> for crate::device::Cubecl {
  fn euler_kernel<P: EulerCoefficients<f32>>(
    &self,
    process: &P,
    first: usize,
    m: usize,
    seed: u64,
  ) -> DeviceResult<Array2<f32>> {
    let spec = process.euler_spec();
    let x0 = process.initial_value();
    let n = process.grid_points();
    let t = process.horizon();
    match self.device {
      #[cfg(feature = "cubecl-cuda")]
      crate::device::CubeclDevice::Cuda => {
        device_paths::<cuda_rt::Rt>(self.ordinal, spec, x0, n, t, first, m, seed)
      }
      #[cfg(feature = "cubecl-wgpu")]
      crate::device::CubeclDevice::Wgpu => {
        device_paths::<wgpu_rt::Rt>(self.ordinal, spec, x0, n, t, first, m, seed)
      }
    }
  }

  fn batch_budget(&self) -> usize {
    self.batch_budget
  }
}

/// The kernel launch for an explicit specification.
#[allow(clippy::too_many_arguments)]
fn device_paths<C: CubeclRuntime>(
  ordinal: usize,
  spec: EulerSpec<f32>,
  x0: f32,
  n: usize,
  t: f32,
  first: usize,
  m: usize,
  seed: u64,
) -> DeviceResult<Array2<f32>> {
  {
    if n == 0 || m == 0 {
      return Ok(Array2::<f32>::zeros((m, n)));
    }
    let (family, params) = spec.encode();
    let params32: Vec<f32> = params.to_vec();
    let dt = t as f64 / (n.max(2) - 1) as f64;
    let total = m * n;
    let cl = &C::client(ordinal)?;
    let data: Vec<f32> = {
      let params_h = cl.create_from_slice(f32::as_bytes(&params32));
      let out_h = cl.empty(total * 4);
      unsafe {
        euler_paths_kernel::launch::<C::Rt>(
          cl,
          count_2d((m as u32).div_ceil(WG_SIZE)),
          CubeDim::new_1d(WG_SIZE),
          ArrayArg::from_raw_parts::<f32>(&out_h, total, 1),
          ArrayArg::from_raw_parts::<f32>(&params_h, 4, 1),
          ScalarArg::new(family),
          ScalarArg::new(x0),
          ScalarArg::new(dt as f32),
          ScalarArg::new(dt.sqrt() as f32),
          ScalarArg::new((seed ^ (seed >> 32)) as u32),
          ScalarArg::new(n as u32),
          ScalarArg::new(m as u32),
          ScalarArg::new(first as u32),
        )
        .map_err(|e| DeviceError::Launch(format!("euler_paths launch: {e:?}")))?;
      }
      let bytes = cl.read_one(out_h.clone());
      f32::from_bytes(&bytes).to_vec()
    };
    Ok(Array2::from_shape_vec((m, n), data).expect("the kernel returns m * n values"))
  }
}
