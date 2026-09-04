//! cudarc + NVRTC device path of the Euler engine: one CUDA thread per path,
//! the whole recursion in the kernel, normals from the same counter hash of
//! `(path, step, seed)` as the CubeCL and Metal kernels (so the three device
//! back-ends agree seed for seed up to libm rounding), in `f32` or `f64`
//! according to `T` — NVIDIA hardware has native double precision.

use std::any::TypeId;
use std::sync::Arc;

use cudarc::driver::*;
use cudarc::nvrtc;
use ndarray::Array2;
use parking_lot::Mutex;

use super::EulerCoefficients;
use super::EulerKernel;
use super::EulerSpec;
use crate::device::Cuda;
use crate::device::DeviceError;
use crate::device::DeviceInfo;
use crate::noise::fgn::cuda::PinnedHost;
use crate::traits::FloatExt;

type Result<T> = std::result::Result<T, DeviceError>;

/// The `float` / `double` kernel: the launch header around the body the
/// Metal back-end renders too ([`super::kernel`]).
const CUDA_HEADER: &str = r#"extern "C" __global__ void euler_paths_REAL(
    REAL* __restrict__ out,
    const REAL* __restrict__ params,
    unsigned int family, REAL x0, REAL dt, REAL sqrt_dt,
    unsigned int seed, unsigned int steps, unsigned int paths,
    unsigned int first_path,
    const REAL* __restrict__ incs, unsigned int increments)
{
    unsigned int path = blockIdx.x * blockDim.x + threadIdx.x;
"#;

fn kernel_source(real: &str) -> String {
  let lang = if real == "float" {
    super::kernel::Language {
      real,
      sqrt: "sqrtf",
      log: "logf",
      cos: "cosf",
      exp: "expf",
      pow: "powf",
      abs: "fabsf",
      index: "unsigned long long",
    }
  } else {
    super::kernel::Language {
      real,
      sqrt: "sqrt",
      log: "log",
      cos: "cos",
      exp: "exp",
      pow: "pow",
      abs: "fabs",
      index: "unsigned long long",
    }
  };
  let prelude = super::kernel::prelude(&lang);
  let body = super::kernel::render(&lang);
  format!("{prelude}{}{body}}}\n", CUDA_HEADER.replace("REAL", real))
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

fn ensure_kernels(ordinal: usize) -> Result<()> {
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
  ordinal: usize,
  func: impl Fn(&Kernels) -> &CudaFunction,
  params: [R; 4],
  x0: R,
  dt: R,
  family: u32,
  seed: u32,
  first: usize,
  n: usize,
  m: usize,
  increments: &[R],
) -> Result<Vec<R>>
where
  R: DeviceRepr + ValidAsZeroBits + Copy + num_traits::Float,
{
  ensure_kernels(ordinal)?;
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
  // The kernel always binds the increment pointer; an unused slot gets one
  // element rather than a null.
  let use_incs = u32::from(!increments.is_empty());
  let d_incs = if increments.is_empty() {
    stream
      .alloc_zeros::<R>(1)
      .map_err(|e| DeviceError::Launch(format!("alloc incs: {e}")))?
  } else {
    stream
      .clone_htod(increments)
      .map_err(|e| DeviceError::Launch(format!("htod incs: {e}")))?
  };
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
      .arg(&d_incs)
      .arg(&use_incs)
      .launch(LaunchConfig::for_num_elems(paths))
      .map_err(|e| DeviceError::Launch(format!("euler_paths: {e}")))?;
  }
  stream
    .clone_dtoh(&d_out)
    .map_err(|e| DeviceError::Launch(format!("dtoh: {e}")))
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
      process.initial_value(),
      process.grid_points(),
      process.horizon(),
      first,
      m,
      seed,
      process.increments(first, m, seed).as_deref().unwrap_or(&[]),
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
    pipelined_paths(
      self.ordinal,
      process.euler_spec(),
      process.initial_value(),
      n,
      process.horizon(),
      m,
      rows,
      seed,
    )
  }
}

/// Re-types a slice of `T` into the precision the kernel runs in. `T` is
/// `f32` or `f64` by `FloatExt`'s own bound, so the conversion is a copy when
/// the precisions differ and a borrow-shaped copy when they do not.
fn cast_slice<T: FloatExt, R: num_traits::Float>(values: &[T]) -> Vec<R> {
  values
    .iter()
    .map(|v| R::from(v.to_f64().unwrap_or(0.0)).unwrap_or_else(R::zero))
    .collect()
}

/// The kernel launch for an explicit specification.
#[allow(clippy::too_many_arguments)]
fn device_paths<T: FloatExt>(
  ordinal: usize,
  spec: EulerSpec<T>,
  x0: T,
  n: usize,
  t: T,
  first: usize,
  m: usize,
  seed: u64,
  increments: &[T],
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
        ordinal,
        |k| &k.f64,
        p64,
        x0.to_f64().unwrap_or(0.0),
        dt,
        family,
        seed32,
        first,
        n,
        m,
        &cast_slice::<T, f64>(increments),
      )?;
      let out =
        Array2::<f64>::from_shape_vec((m, n), data).expect("the kernel returns m * n values");
      return Ok(unsafe { std::mem::transmute::<Array2<f64>, Array2<T>>(out) });
    }
    let p32: [f32; 4] = std::array::from_fn(|i| p64[i] as f32);
    let data = run::<f32>(
      ordinal,
      |k| &k.f32,
      p32,
      x0.to_f64().unwrap_or(0.0) as f32,
      dt as f32,
      family,
      seed32,
      first,
      n,
      m,
      &cast_slice::<T, f32>(increments),
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
  increments: &[R],
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
  // The kernel always binds the increment pointer; an unused slot gets one
  // element rather than a null.
  let use_incs = u32::from(!increments.is_empty());
  let d_incs = if increments.is_empty() {
    stream
      .alloc_zeros::<R>(1)
      .map_err(|e| DeviceError::Launch(format!("alloc incs: {e}")))?
  } else {
    stream
      .clone_htod(increments)
      .map_err(|e| DeviceError::Launch(format!("htod incs: {e}")))?
  };
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
      .arg(&d_incs)
      .arg(&use_incs)
      .launch(LaunchConfig::for_num_elems(paths))
      .map_err(|e| DeviceError::Launch(format!("euler_paths: {e}")))?;
  }
  Ok(d_out)
}

/// The whole batch through the two-stream pipeline, `rows` paths per chunk.
#[allow(clippy::too_many_arguments)]
fn pipelined<R>(
  ordinal: usize,
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
  ensure_kernels(ordinal)?;
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
      // The pipelined batch is Gaussian only: a fractional process reports a
      // batch budget that keeps it to one launch.
      &[],
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
  ordinal: usize,
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
      ordinal,
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
    ordinal,
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
