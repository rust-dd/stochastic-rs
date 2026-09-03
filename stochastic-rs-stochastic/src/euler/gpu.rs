//! CubeCL device path of the Euler engine: one thread per path, the whole
//! recursion in the kernel, normals from a counter hash of
//! `(path, step, seed)` pushed through Box–Muller — the same generator as
//! the fGN device kernels. `f32` on the device (the portable GPU float),
//! widened on the way back.

use cubecl::prelude::*;
use ndarray::Array1;
use ndarray::Array2;
use parking_lot::Mutex;

use super::EulerBackend;
use super::EulerCoefficients;
use super::EulerSpec;
use crate::device::CubeCl;
use crate::device::DeviceError;
use crate::device::DeviceInfo;

type DeviceResult<T> = std::result::Result<T, DeviceError>;

#[cfg(feature = "cubecl-cuda")]
type R = cubecl_cuda::CudaRuntime;
#[cfg(all(feature = "cubecl-wgpu", not(feature = "cubecl-cuda")))]
type R = cubecl_wgpu::WgpuRuntime;

const WG_SIZE: u32 = 256;

#[cube(launch)]
fn euler_paths_kernel<F: Float + CubeElement>(
  out: &mut Array<F>,
  params: &Array<F>,
  family: u32,
  x0: F,
  dt: F,
  sqrt_dt: F,
  seed: u32,
  steps: u32,
  paths: u32,
) {
  let path = ABSOLUTE_POS as u32;
  if path < paths {
    let base = (path * steps) as usize;
    let mut x = x0;
    let mut reported = x0;
    if family == 2u32 && x0 < F::new(0.0_f32) {
      reported = F::new(0.0_f32);
    }
    out[base] = reported;
    for i in 1..steps {
      // Two decorrelated uniforms via integer hashing (Murmur3-style finalizer).
      let g = path * steps + i;
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
      let inv = F::new(2.3283064e-10_f32);
      let u1 = F::cast_from(a) * inv * F::new(0.999998_f32) + F::new(1.0e-6_f32);
      let u2 = F::cast_from(b) * inv;
      let z = F::sqrt(F::new(-2.0_f32) * F::ln(u1)) * F::cos(F::new(core::f32::consts::TAU) * u2);
      if family == 0u32 {
        x += params[0] * x * dt + params[1] * x * sqrt_dt * z;
      } else if family == 1u32 {
        x += params[0] * (params[1] - x) * dt + params[2] * sqrt_dt * z;
      } else {
        let mut positive = F::new(0.0_f32);
        if x > F::new(0.0_f32) {
          positive = x;
        }
        x =
          x + params[0] * (params[1] - positive) * dt + params[2] * F::sqrt(positive) * sqrt_dt * z;
      }
      let mut reported = x;
      if family == 2u32 && x < F::new(0.0_f32) {
        reported = F::new(0.0_f32);
      }
      out[base + i as usize] = reported;
    }
  }
}

struct Context {
  ordinal: usize,
  client: cubecl::client::ComputeClient<R>,
}

unsafe impl Send for Context {}

static CONTEXT: Mutex<Option<Context>> = Mutex::new(None);

/// The cached compute client, opened on first use. CubeCL panics rather than
/// erroring when no device exists, so the opening is caught and reported.
fn client() -> DeviceResult<cubecl::client::ComputeClient<R>> {
  let ordinal = crate::device::selected_device();
  let mut guard = CONTEXT.lock();
  if !guard.as_ref().is_some_and(|c| c.ordinal == ordinal) {
    *guard = None;
    let opened = std::panic::catch_unwind(|| R::client(&selected_cubecl_device()));
    match opened {
      Ok(client) => *guard = Some(Context { ordinal, client }),
      Err(payload) => {
        return Err(DeviceError::Unavailable(crate::device::panic_text(payload)));
      }
    }
  }
  Ok(guard.as_ref().expect("initialised").client.clone())
}

/// The CubeCL device [`crate::device::select_device`] chose.
pub(crate) fn selected_cubecl_device() -> <R as cubecl::Runtime>::Device {
  let ordinal = crate::device::selected_device();
  #[cfg(feature = "cubecl-cuda")]
  {
    cubecl_cuda::CudaDevice { index: ordinal }
  }

  #[cfg(all(feature = "cubecl-wgpu", not(feature = "cubecl-cuda")))]
  {
    if ordinal == 0 {
      cubecl_wgpu::WgpuDevice::default()
    } else {
      cubecl_wgpu::WgpuDevice::DiscreteGpu(ordinal)
    }
  }
}

/// The selected CubeCL device, or why it cannot be used.
pub(crate) fn probe() -> DeviceResult<DeviceInfo> {
  let cl = client()?;
  Ok(DeviceInfo::new(
    "CubeCl",
    <R as cubecl::Runtime>::name(&cl).to_string(),
    &["f32"],
    Some(crate::device::selected_device()),
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

impl EulerBackend<f32> for CubeCl {
  const DEVICE: bool = true;

  fn try_euler_paths<P: EulerCoefficients<f32>>(
    process: &P,
    m: usize,
  ) -> DeviceResult<Vec<Array1<f32>>> {
    Ok(
      device_paths(
        process.euler_spec(),
        process.initial_value(),
        process.grid_points(),
        process.horizon(),
        m,
        process.device_seed(),
      )?
      .outer_iter()
      .map(|row| row.to_owned())
      .collect(),
    )
  }
}

/// The kernel launch for an explicit specification.
fn device_paths(
  spec: EulerSpec<f32>,
  x0: f32,
  n: usize,
  t: f32,
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
    let cl = &client()?;
    let data: Vec<f32> = {
      let params_h = cl.create_from_slice(f32::as_bytes(&params32));
      let out_h = cl.empty(total * 4);
      unsafe {
        euler_paths_kernel::launch::<f32, R>(
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
        )
        .map_err(|e| DeviceError::Launch(format!("euler_paths launch: {e:?}")))?;
      }
      let bytes = cl.read_one(out_h.clone());
      f32::from_bytes(&bytes).to_vec()
    };
    Ok(Array2::from_shape_vec((m, n), data).expect("the kernel returns m * n values"))
  }
}
