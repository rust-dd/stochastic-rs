//! CubeCL device path of the Euler engine: one thread per path, the whole
//! recursion in the kernel, normals from a counter hash of
//! `(path, step, seed)` pushed through Box–Muller — the same generator as
//! the fGN device kernels. `f32` on the device (the portable GPU float),
//! widened on the way back.

use cubecl::prelude::*;
use ndarray::Array2;
use parking_lot::Mutex;

use super::EulerBackend;
use super::EulerSpec;
use crate::device::CubeCl;
use crate::traits::FloatExt;

#[cfg(feature = "gpu-cuda")]
type R = cubecl_cuda::CudaRuntime;
#[cfg(all(feature = "gpu-wgpu", not(feature = "gpu-cuda")))]
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
  client: cubecl::client::ComputeClient<R>,
}

unsafe impl Send for Context {}

static CONTEXT: Mutex<Option<Context>> = Mutex::new(None);

fn with_client<Out>(f: impl FnOnce(&cubecl::client::ComputeClient<R>) -> Out) -> Out {
  let mut guard = CONTEXT.lock();
  if guard.is_none() {
    #[cfg(feature = "gpu-cuda")]
    let device = cubecl_cuda::CudaDevice::default();
    #[cfg(all(feature = "gpu-wgpu", not(feature = "gpu-cuda")))]
    let device = cubecl_wgpu::WgpuDevice::default();
    *guard = Some(Context {
      client: R::client(&device),
    });
  }
  f(&guard.as_ref().expect("initialised").client)
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

impl EulerBackend for CubeCl {
  fn euler_paths<T: FloatExt>(
    spec: EulerSpec<T>,
    x0: T,
    n: usize,
    t: T,
    m: usize,
    seed: u64,
  ) -> Array2<T> {
    if n == 0 || m == 0 {
      return Array2::<T>::zeros((m, n));
    }
    let (family, params) = spec.encode();
    let params32: Vec<f32> = params
      .iter()
      .map(|p| p.to_f64().unwrap_or(0.0) as f32)
      .collect();
    let dt = t.to_f64().unwrap_or(1.0) / (n.max(2) - 1) as f64;
    let total = m * n;
    let data: Vec<f32> = with_client(|cl| {
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
          ScalarArg::new(x0.to_f64().unwrap_or(0.0) as f32),
          ScalarArg::new(dt as f32),
          ScalarArg::new(dt.sqrt() as f32),
          ScalarArg::new((seed ^ (seed >> 32)) as u32),
          ScalarArg::new(n as u32),
          ScalarArg::new(m as u32),
        )
        .expect("Euler engine kernel launch");
      }
      let bytes = cl.read_one(out_h.clone());
      f32::from_bytes(&bytes).to_vec()
    });
    let mut out = Array2::<T>::zeros((m, n));
    for i in 0..m {
      for j in 0..n {
        out[[i, j]] = T::from_f64_fast(data[i * n + j] as f64);
      }
    }
    out
  }
}
