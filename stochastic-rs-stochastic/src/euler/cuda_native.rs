//! cudarc + NVRTC device path of the Euler engine: one CUDA thread per path,
//! the whole recursion in the kernel, normals from the same counter hash of
//! `(path, step, seed)` as the CubeCL and Metal kernels (so the three device
//! back-ends agree seed for seed up to libm rounding), in `f32` or `f64`
//! according to `T` — NVIDIA hardware has native double precision.

use std::any::TypeId;
use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::*;
use cudarc::nvrtc;
use ndarray::Array1;
use ndarray::Array2;
use parking_lot::Mutex;

use super::EulerBackend;
use super::EulerCoefficients;
use super::EulerSpec;
use crate::device::CudaNative;
use crate::traits::FloatExt;

/// The kernel body shared by both precisions; `REAL` is substituted at
/// compile time.
const KERNEL_TEMPLATE: &str = r#"
extern "C" __global__ void euler_paths_REAL(
    REAL* __restrict__ out,
    const REAL* __restrict__ params,
    unsigned int family, REAL x0, REAL dt, REAL sqrt_dt,
    unsigned int seed, unsigned int steps, unsigned int paths)
{
    unsigned int path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= paths) return;
    unsigned long long base = (unsigned long long)path * steps;
    REAL x = x0;
    REAL reported = x0;
    if (family == 2u && x0 < (REAL)0) reported = (REAL)0;
    out[base] = reported;
    for (unsigned int i = 1; i < steps; i++) {
        unsigned int g = path * steps + i;
        unsigned int a = (g * 2u) ^ (seed * 2654435761u);
        a ^= a >> 16; a *= 2246822519u; a ^= a >> 13; a *= 3266489917u; a ^= a >> 16;
        unsigned int b = (g * 2u + 1u) ^ (seed * 668265263u);
        b ^= b >> 16; b *= 2246822519u; b ^= b >> 13; b *= 3266489917u; b ^= b >> 16;
        REAL u1 = (REAL)a * (REAL)2.3283064e-10 * (REAL)0.999998 + (REAL)1.0e-6;
        REAL u2 = (REAL)b * (REAL)2.3283064e-10;
        REAL z = SQRT((REAL)-2.0 * LOG(u1)) * COS((REAL)6.283185307179586 * u2);
        if (family == 0u) {
            x = x + params[0] * x * dt + params[1] * x * sqrt_dt * z;
        } else if (family == 1u) {
            x = x + params[0] * (params[1] - x) * dt + params[2] * sqrt_dt * z;
        } else {
            REAL positive = x > (REAL)0 ? x : (REAL)0;
            x = x + params[0] * (params[1] - positive) * dt + params[2] * SQRT(positive) * sqrt_dt * z;
        }
        reported = x;
        if (family == 2u && x < (REAL)0) reported = (REAL)0;
        out[base + i] = reported;
    }
}
"#;

fn kernel_source(real: &str) -> String {
  let (sqrt, log, cos) = if real == "float" {
    ("sqrtf", "logf", "cosf")
  } else {
    ("sqrt", "log", "cos")
  };
  KERNEL_TEMPLATE
    .replace("SQRT", sqrt)
    .replace("LOG", log)
    .replace("COS", cos)
    .replace("REAL", real)
}

struct Kernels {
  stream: Arc<CudaStream>,
  f32: CudaFunction,
  f64: CudaFunction,
}

/// SAFETY: every device operation is serialised through the one stream.
unsafe impl Send for Kernels {}

static KERNELS: Mutex<Option<Kernels>> = Mutex::new(None);

fn ensure_kernels() -> Result<()> {
  let mut guard = KERNELS.lock();
  if guard.is_some() {
    return Ok(());
  }
  let ctx = CudaContext::new(0).map_err(|e| anyhow::anyhow!("CudaContext: {e}"))?;
  let stream = ctx
    .new_stream()
    .map_err(|e| anyhow::anyhow!("stream: {e}"))?;
  let context = stream.context();
  let load = |real: &str| -> Result<CudaFunction> {
    let src = kernel_source(real);
    let name = format!("euler_paths_{real}");
    let ptx = nvrtc::compile_ptx(src).map_err(|e| anyhow::anyhow!("NVRTC {name}: {e}"))?;
    let module = context
      .load_module(ptx)
      .map_err(|e| anyhow::anyhow!("load {name}: {e}"))?;
    module
      .load_function(&name)
      .map_err(|e| anyhow::anyhow!("fn {name}: {e}"))
  };
  *guard = Some(Kernels {
    f32: load("float")?,
    f64: load("double")?,
    stream,
  });
  Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run<R>(
  func: impl Fn(&Kernels) -> &CudaFunction,
  params: [R; 4],
  x0: R,
  dt: R,
  family: u32,
  seed: u32,
  n: usize,
  m: usize,
) -> Result<Vec<R>>
where
  R: DeviceRepr + ValidAsZeroBits + Copy + num_traits::Float,
{
  ensure_kernels()?;
  let guard = KERNELS.lock();
  let kernels = guard.as_ref().expect("initialised");
  let stream = &kernels.stream;
  let d_params = stream
    .clone_htod(&params[..])
    .map_err(|e| anyhow::anyhow!("htod params: {e}"))?;
  let mut d_out = stream
    .alloc_zeros::<R>(m * n)
    .map_err(|e| anyhow::anyhow!("alloc out: {e}"))?;
  let sqrt_dt = dt.sqrt();
  let (steps, paths) = (n as u32, m as u32);
  unsafe {
    stream
      .launch_builder(func(kernels))
      .arg(&mut d_out)
      .arg(&d_params)
      .arg(&family)
      .arg(&x0)
      .arg(&dt)
      .arg(&sqrt_dt)
      .arg(&seed)
      .arg(&steps)
      .arg(&paths)
      .launch(LaunchConfig::for_num_elems(paths))
      .map_err(|e| anyhow::anyhow!("euler_paths: {e}"))?;
  }
  stream
    .clone_dtoh(&d_out)
    .map_err(|e| anyhow::anyhow!("dtoh: {e}"))
}

impl<T: FloatExt> EulerBackend<T> for CudaNative {
  const DEVICE: bool = true;

  fn euler_paths<P: EulerCoefficients<T>>(process: &P, m: usize) -> Vec<Array1<T>> {
    device_paths(
      process.euler_spec(),
      process.initial_value(),
      process.grid_points(),
      process.horizon(),
      m,
      process.device_seed(),
    )
    .outer_iter()
    .map(|row| row.to_owned())
    .collect()
  }
}

/// The kernel launch for an explicit specification.
fn device_paths<T: FloatExt>(
  spec: EulerSpec<T>,
  x0: T,
  n: usize,
  t: T,
  m: usize,
  seed: u64,
) -> Array2<T> {
  {
    if n == 0 || m == 0 {
      return Array2::<T>::zeros((m, n));
    }
    let (family, params) = spec.encode();
    let dt = t.to_f64().unwrap_or(1.0) / (n.max(2) - 1) as f64;
    let seed32 = (seed ^ (seed >> 32)) as u32;
    let p64: [f64; 4] = std::array::from_fn(|i| params[i].to_f64().unwrap_or(0.0));
    if TypeId::of::<T>() == TypeId::of::<f64>() {
      let data = run::<f64>(
        |k| &k.f64,
        p64,
        x0.to_f64().unwrap_or(0.0),
        dt,
        family,
        seed32,
        n,
        m,
      )
      .expect("native CUDA Euler engine");
      let out =
        Array2::<f64>::from_shape_vec((m, n), data).expect("the kernel returns m * n values");
      return unsafe { std::mem::transmute::<Array2<f64>, Array2<T>>(out) };
    }
    let p32: [f32; 4] = std::array::from_fn(|i| p64[i] as f32);
    let data = run::<f32>(
      |k| &k.f32,
      p32,
      x0.to_f64().unwrap_or(0.0) as f32,
      dt as f32,
      family,
      seed32,
      n,
      m,
    )
    .expect("native CUDA Euler engine");
    assert!(
      TypeId::of::<T>() == TypeId::of::<f32>(),
      "FloatExt is implemented for f32 and f64 only"
    );
    let out = Array2::<f32>::from_shape_vec((m, n), data).expect("the kernel returns m * n values");
    unsafe { std::mem::transmute::<Array2<f32>, Array2<T>>(out) }
  }
}
