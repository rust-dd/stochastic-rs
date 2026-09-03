//! cudarc + NVRTC device path of the Euler engine: one CUDA thread per path,
//! the whole recursion in the kernel, normals from the same counter hash of
//! `(path, step, seed)` as the CubeCL and Metal kernels (so the three device
//! back-ends agree seed for seed up to libm rounding), in `f32` or `f64`
//! according to `T` — NVIDIA hardware has native double precision.

use std::any::TypeId;
use std::sync::Arc;

use cudarc::driver::*;
use cudarc::nvrtc;
use ndarray::Array1;
use ndarray::Array2;
use parking_lot::Mutex;

use super::EulerBackend;
use super::EulerCoefficients;
use super::EulerSpec;
use crate::device::CudaNative;
use crate::device::DeviceError;
use crate::device::DeviceInfo;
use crate::noise::fgn::cuda_native::PinnedHost;
use crate::traits::FloatExt;

type Result<T> = std::result::Result<T, DeviceError>;

/// The kernel body shared by both precisions; `REAL` is substituted at
/// compile time.
const KERNEL_TEMPLATE: &str = r#"
extern "C" __global__ void euler_paths_REAL(
    REAL* __restrict__ out,
    const REAL* __restrict__ params,
    unsigned int family, REAL x0, REAL dt, REAL sqrt_dt,
    unsigned int seed, unsigned int steps, unsigned int paths,
    unsigned int first_path)
{
    unsigned int path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= paths) return;
    unsigned long long base = (unsigned long long)path * steps;
    REAL x = x0;
    REAL reported = x0;
    if (family == 2u && x0 < (REAL)0) reported = (REAL)0;
    out[base] = reported;
    for (unsigned int i = 1; i < steps; i++) {
        unsigned int g = (first_path + path) * steps + i;
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
  ordinal: usize,
  stream: Arc<CudaStream>,
  /// Second stream of the batch pipeline: chunk `k + 1` computes on one
  /// while chunk `k` copies back on the other.
  stream_b: Arc<CudaStream>,
  f32: CudaFunction,
  f64: CudaFunction,
}

/// SAFETY: every device operation is serialised through the one stream.
unsafe impl Send for Kernels {}

static KERNELS: Mutex<Option<Kernels>> = Mutex::new(None);

/// The selected CUDA device, or why it cannot be used.
pub(crate) fn probe() -> Result<DeviceInfo> {
  let ordinal = crate::device::selected_device();
  let ctx =
    CudaContext::new(ordinal).map_err(|e| DeviceError::Unavailable(format!("CudaContext: {e}")))?;
  let name = ctx
    .name()
    .map_err(|e| DeviceError::Unavailable(format!("device name: {e}")))?;
  Ok(DeviceInfo::new(
    "CudaNative",
    name,
    &["f32", "f64"],
    Some(ordinal),
  ))
}

fn ensure_kernels() -> Result<()> {
  let ordinal = crate::device::selected_device();
  let mut guard = KERNELS.lock();
  if guard.as_ref().is_some_and(|k| k.ordinal == ordinal) {
    return Ok(());
  }
  *guard = None;
  let ctx =
    CudaContext::new(ordinal).map_err(|e| DeviceError::Unavailable(format!("CudaContext: {e}")))?;
  let stream = ctx
    .new_stream()
    .map_err(|e| DeviceError::Launch(format!("stream: {e}")))?;
  let context = stream.context();
  let load = |real: &str| -> Result<CudaFunction> {
    let src = kernel_source(real);
    let name = format!("euler_paths_{real}");
    let ptx =
      nvrtc::compile_ptx(src).map_err(|e| DeviceError::Compile(format!("NVRTC {name}: {e}")))?;
    let module = context
      .load_module(ptx)
      .map_err(|e| DeviceError::Launch(format!("load {name}: {e}")))?;
    module
      .load_function(&name)
      .map_err(|e| DeviceError::Launch(format!("fn {name}: {e}")))
  };
  let stream_b = ctx
    .new_stream()
    .map_err(|e| DeviceError::Launch(format!("stream: {e}")))?;
  *guard = Some(Kernels {
    ordinal,
    f32: load("float")?,
    f64: load("double")?,
    stream,
    stream_b,
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
  first: usize,
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
    .map_err(|e| DeviceError::Launch(format!("htod params: {e}")))?;
  let mut d_out = stream
    .alloc_zeros::<R>(m * n)
    .map_err(|e| DeviceError::Launch(format!("alloc out: {e}")))?;
  let sqrt_dt = dt.sqrt();
  let (steps, paths, first_path) = (n as u32, m as u32, first as u32);
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
      .arg(&first_path)
      .launch(LaunchConfig::for_num_elems(paths))
      .map_err(|e| DeviceError::Launch(format!("euler_paths: {e}")))?;
  }
  stream
    .clone_dtoh(&d_out)
    .map_err(|e| DeviceError::Launch(format!("dtoh: {e}")))
}

impl<T: FloatExt> EulerBackend<T> for CudaNative {
  const DEVICE: bool = true;

  /// Chunks alternate between two streams: while chunk `k` copies back
  /// through pinned memory, chunk `k + 1` is already computing. The union is
  /// bit-identical to one launch (the kernel hashes the global path index).
  fn try_euler_paths<P: EulerCoefficients<T>>(process: &P, m: usize) -> Result<Vec<Array1<T>>> {
    let n = process.grid_points();
    let rows = crate::device::chunk_rows(n, std::mem::size_of::<T>());
    if m <= rows {
      return Self::try_euler_paths_from(process, 0, m);
    }
    Ok(
      pipelined_paths(
        process.euler_spec(),
        process.initial_value(),
        n,
        process.horizon(),
        m,
        rows,
        process.device_seed(),
      )?
      .outer_iter()
      .map(|row| row.to_owned())
      .collect(),
    )
  }

  /// The launch buffer as the matrix, chunked like the row form when the
  /// batch exceeds the budget.
  fn try_euler_matrix<P: EulerCoefficients<T>>(process: &P, m: usize) -> Result<Array2<T>> {
    let n = process.grid_points();
    let seed = process.device_seed();
    let rows = crate::device::chunk_rows(n, std::mem::size_of::<T>());
    if m <= rows {
      return device_paths(
        process.euler_spec(),
        process.initial_value(),
        n,
        process.horizon(),
        0,
        m,
        seed,
      );
    }
    pipelined_paths(
      process.euler_spec(),
      process.initial_value(),
      n,
      process.horizon(),
      m,
      rows,
      seed,
    )
  }

  fn try_euler_paths_seeded<P: EulerCoefficients<T>>(
    process: &P,
    first: usize,
    m: usize,
    seed: u64,
  ) -> Result<Vec<Array1<T>>> {
    Ok(
      device_paths(
        process.euler_spec(),
        process.initial_value(),
        process.grid_points(),
        process.horizon(),
        first,
        m,
        seed,
      )?
      .outer_iter()
      .map(|row| row.to_owned())
      .collect(),
    )
  }
}

/// The kernel launch for an explicit specification.
fn device_paths<T: FloatExt>(
  spec: EulerSpec<T>,
  x0: T,
  n: usize,
  t: T,
  first: usize,
  m: usize,
  seed: u64,
) -> Result<Array2<T>> {
  {
    if n == 0 || m == 0 {
      return Ok(Array2::<T>::zeros((m, n)));
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
        first,
        n,
        m,
      )?;
      let out =
        Array2::<f64>::from_shape_vec((m, n), data).expect("the kernel returns m * n values");
      return Ok(unsafe { std::mem::transmute::<Array2<f64>, Array2<T>>(out) });
    }
    let p32: [f32; 4] = std::array::from_fn(|i| p64[i] as f32);
    let data = run::<f32>(
      |k| &k.f32,
      p32,
      x0.to_f64().unwrap_or(0.0) as f32,
      dt as f32,
      family,
      seed32,
      first,
      n,
      m,
    )?;
    assert!(
      TypeId::of::<T>() == TypeId::of::<f32>(),
      "FloatExt is implemented for f32 and f64 only"
    );
    let out = Array2::<f32>::from_shape_vec((m, n), data).expect("the kernel returns m * n values");
    Ok(unsafe { std::mem::transmute::<Array2<f32>, Array2<T>>(out) })
  }
}

/// One chunk's launch without the copy back: the caller owns the stream.
#[allow(clippy::too_many_arguments)]
fn launch_chunk<R>(
  stream: &Arc<CudaStream>,
  func: &CudaFunction,
  params: [R; 4],
  x0: R,
  dt: R,
  family: u32,
  seed: u32,
  first: usize,
  n: usize,
  m: usize,
) -> Result<CudaSlice<R>>
where
  R: DeviceRepr + ValidAsZeroBits + Copy + num_traits::Float,
{
  let d_params = stream
    .clone_htod(&params[..])
    .map_err(|e| DeviceError::Launch(format!("htod params: {e}")))?;
  let mut d_out = stream
    .alloc_zeros::<R>(m * n)
    .map_err(|e| DeviceError::Launch(format!("alloc out: {e}")))?;
  let sqrt_dt = dt.sqrt();
  let (steps, paths, first_path) = (n as u32, m as u32, first as u32);
  unsafe {
    stream
      .launch_builder(func)
      .arg(&mut d_out)
      .arg(&d_params)
      .arg(&family)
      .arg(&x0)
      .arg(&dt)
      .arg(&sqrt_dt)
      .arg(&seed)
      .arg(&steps)
      .arg(&paths)
      .arg(&first_path)
      .launch(LaunchConfig::for_num_elems(paths))
      .map_err(|e| DeviceError::Launch(format!("euler_paths: {e}")))?;
  }
  Ok(d_out)
}

/// The whole batch through the two-stream pipeline, `rows` paths per chunk.
#[allow(clippy::too_many_arguments)]
fn pipelined<R>(
  func: impl Fn(&Kernels) -> &CudaFunction,
  params: [R; 4],
  x0: R,
  dt: R,
  family: u32,
  seed: u32,
  n: usize,
  m: usize,
  rows: usize,
) -> Result<Vec<R>>
where
  R: DeviceRepr + ValidAsZeroBits + Copy + num_traits::Float + Send + Sync,
{
  ensure_kernels()?;
  let guard = KERNELS.lock();
  let kernels = guard.as_ref().expect("initialised");
  let streams = [kernels.stream.clone(), kernels.stream_b.clone()];
  let func = func(kernels).clone();
  drop(guard);
  let mut host = vec![R::zero(); m * n];
  let staging = [
    PinnedHost::<R>::alloc(rows * n)?,
    PinnedHost::<R>::alloc(rows * n)?,
  ];
  // Per slot: the device buffer kept alive until its copy has landed, and
  // the `(first, len)` rows the staging buffer holds.
  let mut in_flight: [Option<(CudaSlice<R>, usize, usize)>; 2] = [None, None];
  let drain = |slot: usize,
               in_flight: &mut [Option<(CudaSlice<R>, usize, usize)>; 2],
               host: &mut [R]|
   -> Result<()> {
    if let Some((_d_out, f0, l0)) = in_flight[slot].take() {
      streams[slot]
        .synchronize()
        .map_err(|e| DeviceError::Launch(format!("sync chunk: {e}")))?;
      let src = unsafe { std::slice::from_raw_parts(staging[slot].ptr, l0 * n) };
      host[f0 * n..(f0 + l0) * n].copy_from_slice(src);
    }
    Ok(())
  };
  let mut first = 0;
  let mut k = 0;
  while first < m {
    let len = rows.min(m - first);
    let slot = k % 2;
    drain(slot, &mut in_flight, &mut host)?;
    let d_out = launch_chunk(
      &streams[slot],
      &func,
      params,
      x0,
      dt,
      family,
      seed,
      first,
      n,
      len,
    )?;
    let dst = unsafe { std::slice::from_raw_parts_mut(staging[slot].ptr, len * n) };
    streams[slot]
      .memcpy_dtoh(&d_out, dst)
      .map_err(|e| DeviceError::Launch(format!("dtoh chunk: {e}")))?;
    in_flight[slot] = Some((d_out, first, len));
    first += len;
    k += 1;
  }
  drain(0, &mut in_flight, &mut host)?;
  drain(1, &mut in_flight, &mut host)?;
  Ok(host)
}

/// The pipelined batch for an explicit specification, in the precision of `T`.
#[allow(clippy::too_many_arguments)]
fn pipelined_paths<T: FloatExt>(
  spec: EulerSpec<T>,
  x0: T,
  n: usize,
  t: T,
  m: usize,
  rows: usize,
  seed: u64,
) -> Result<Array2<T>> {
  let (family, params) = spec.encode();
  let dt = t.to_f64().unwrap_or(1.0) / (n.max(2) - 1) as f64;
  let seed32 = (seed ^ (seed >> 32)) as u32;
  let p64: [f64; 4] = std::array::from_fn(|i| params[i].to_f64().unwrap_or(0.0));
  if TypeId::of::<T>() == TypeId::of::<f64>() {
    let data = pipelined::<f64>(
      |k| &k.f64,
      p64,
      x0.to_f64().unwrap_or(0.0),
      dt,
      family,
      seed32,
      n,
      m,
      rows,
    )?;
    let out = Array2::<f64>::from_shape_vec((m, n), data).expect("the kernel returns m * n values");
    return Ok(unsafe { std::mem::transmute::<Array2<f64>, Array2<T>>(out) });
  }
  assert!(
    TypeId::of::<T>() == TypeId::of::<f32>(),
    "FloatExt is implemented for f32 and f64 only"
  );
  let p32: [f32; 4] = std::array::from_fn(|i| p64[i] as f32);
  let data = pipelined::<f32>(
    |k| &k.f32,
    p32,
    x0.to_f64().unwrap_or(0.0) as f32,
    dt as f32,
    family,
    seed32,
    n,
    m,
    rows,
  )?;
  let out = Array2::<f32>::from_shape_vec((m, n), data).expect("the kernel returns m * n values");
  Ok(unsafe { std::mem::transmute::<Array2<f32>, Array2<T>>(out) })
}
