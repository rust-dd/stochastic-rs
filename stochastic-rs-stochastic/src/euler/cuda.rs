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
use ndarray::Array3;
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
    REAL g2_shape, REAL g2_scale, REAL g2_per)
{
    unsigned int path = blockIdx.x * blockDim.x + threadIdx.x;
    const REAL x0[4] = { x00, x01, x02, x03 };
"#;

fn kernel_source(real: &'static str) -> String {
  let lang = super::kernel::cuda_language(real);
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
  let load = |real: &'static str| -> Result<CudaFunction> {
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
    .alloc_zeros::<R>(components as usize * m * n)
    .map_err(|e| DeviceError::Launch(format!("alloc out: {e}")))?;
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
        .map_err(|e| DeviceError::Launch(format!("alloc incs: {e}")))?;
      &owned
    }
  };
  // The kernel always binds the curve pointer; an unused slot gets one
  // element rather than a null.
  let use_curve = n_curves;
  let d_curve = if curve.is_empty() {
    stream
      .alloc_zeros::<R>(1)
      .map_err(|e| DeviceError::Launch(format!("alloc curve: {e}")))?
  } else {
    stream
      .clone_htod(curve)
      .map_err(|e| DeviceError::Launch(format!("htod curve: {e}")))?
  };
  unsafe {
    stream
      .launch_builder(func(kernels))
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
    let planes = device_paths(
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
    )?;
    Ok(planes.index_axis_move(ndarray::Axis(0), 0))
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
    )?;
    Ok(planes.index_axis_move(ndarray::Axis(0), 0))
  }
}

/// The kernel launch for an explicit specification.
#[allow(clippy::too_many_arguments)]
fn device_paths<T: FloatExt>(
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
) -> Result<Array3<T>> {
  {
    let (curve, n_curves) = crate::euler::flatten_curves(curves, n);
    let (family, params) = spec.encode();
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
    if n == 0 || m == 0 {
      return Ok(Array3::<T>::zeros((planes, m, n)));
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
      let data = run::<f64>(
        ordinal,
        |k| &k.f64,
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
      )?;
      let out = Array3::<f64>::from_shape_vec((planes, m, n), data)
        .expect("the kernel returns components * m * n values");
      return Ok(unsafe { std::mem::transmute::<Array3<f64>, Array3<T>>(out) });
    }
    let p32: [f32; crate::euler::PARAM_SLOTS] = std::array::from_fn(|i| p64[i] as f32);
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
    let data = run::<f32>(
      ordinal,
      |k| &k.f32,
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
    )?;
    assert!(
      TypeId::of::<T>() == TypeId::of::<f32>(),
      "FloatExt is implemented for f32 and f64 only"
    );
    let out = Array3::<f32>::from_shape_vec((planes, m, n), data)
      .expect("the kernel returns components * m * n values");
    Ok(unsafe { std::mem::transmute::<Array3<f32>, Array3<T>>(out) })
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
) -> Result<CudaSlice<R>>
where
  R: DeviceRepr + ValidAsZeroBits + Copy + num_traits::Float,
{
  let d_params = stream
    .clone_htod(&params[..])
    .map_err(|e| DeviceError::Launch(format!("htod params: {e}")))?;
  let mut d_out = stream
    .alloc_zeros::<R>(components as usize * m * n)
    .map_err(|e| DeviceError::Launch(format!("alloc out: {e}")))?;
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
        .map_err(|e| DeviceError::Launch(format!("alloc incs: {e}")))?;
      &owned
    }
  };
  // The kernel always binds the curve pointer; an unused slot gets one
  // element rather than a null.
  let use_curve = n_curves;
  let d_curve = if curve.is_empty() {
    stream
      .alloc_zeros::<R>(1)
      .map_err(|e| DeviceError::Launch(format!("alloc curve: {e}")))?
  } else {
    stream
      .clone_htod(curve)
      .map_err(|e| DeviceError::Launch(format!("htod curve: {e}")))?
  };
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
  let planes = components as usize;
  let mut host = vec![R::zero(); planes * m * n];
  let staging = [
    PinnedHost::<R>::alloc(planes * rows * n)?,
    PinnedHost::<R>::alloc(planes * rows * n)?,
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
      components,
      noises,
      seed,
      first,
      n,
      len,
      // The pipelined batch is Gaussian only: a fractional process is launched
      // in one go, so its increments never meet this path.
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
    )?;
    let dst = unsafe { std::slice::from_raw_parts_mut(staging[slot].ptr, planes * len * n) };
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
) -> Result<Array3<T>> {
  let (curve, n_curves) = crate::euler::flatten_curves(curves, n);
  let (family, params) = spec.encode();
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
      |k| &k.f64,
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
    |k| &k.f32,
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
  )?;
  let out = Array3::<f32>::from_shape_vec((planes, m, n), data)
    .expect("the kernel returns components * m * n values");
  Ok(unsafe { std::mem::transmute::<Array3<f32>, Array3<T>>(out) })
}
