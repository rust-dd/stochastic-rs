//! # Euler engine
//!
//! $$
//! X_{i+1} = X_i + b(X_i)\,\Delta t + \sigma(X_i)\sqrt{\Delta t}\,Z_i,\qquad
//! Z_i \sim \mathcal N(0, 1)\ \text{i.i.d.}
//! $$
//!
//! Device-side path generation for the diffusions whose coefficients are a
//! handful of scalars. The backend is a type parameter of the process, as
//! for the fGN-driven types: `Gbm<T, S, B = Cpu>`, `Ou`, `Cir` are switched
//! with `.on::<B2>()` and then sampled through [`ProcessExt`] as usual —
//! `gbm.on::<MetalNative>().sample_par(m)`.
//!
//! - [`Cpu`] (and `Accelerate`, a CPU device) is **the process's own
//!   sampler**, so nothing is re-implemented on the host: GBM keeps its exact
//!   log-normal scheme, OU and CIR their SIMD Euler steppers.
//! - The GPU back-ends run one device thread per path with the whole
//!   Euler–Maruyama recursion in the kernel and Box–Muller normals from a
//!   counter hash of `(path, step, seed)`: `CubeCl` (features `gpu-cuda` /
//!   `gpu-wgpu`: CUDA, Metal, Vulkan or WebGPU through CubeCL, `f32`),
//!   `CudaNative` (feature `cuda-native`: cudarc + NVRTC, `f32` or `f64` after
//!   `T`) and `MetalNative` (feature `metal`: hand-written MSL, `f32`).
//!   `sample_par` is one launch for all `m` paths; `sample` launches one path.
//!
//! The device seed is drawn from the process's own seed source, so a
//! `Deterministic` process reproduces its device paths call after call and an
//! `Unseeded` one draws fresh entropy, exactly as on the host. The device
//! kernels share one integer hash, so the device back-ends agree with each
//! other seed for seed up to libm rounding; the host path is the process's
//! own stream, so CPU and device paths agree in distribution, not bit for bit.
//!
//! A process joins the engine by describing its coefficients as an
//! [`EulerSpec`] through [`EulerCoefficients`].
//!
//! References: Kloeden, P. E. & Platen, E. (1992), *Numerical Solution of
//! Stochastic Differential Equations*, Springer, §10.2 (Euler–Maruyama);
//! Lord, R., Koekkoek, R. & van Dijk, D. (2010), *A comparison of biased
//! simulation schemes for stochastic volatility models*, Quantitative Finance
//! 10(2), 177–194 (full truncation, used by the device kernels for CIR).

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;

use crate::device::Backend;
use crate::device::Cpu;
use crate::device::DeviceError;
use crate::diffusion::cir::Cir;
use crate::diffusion::gbm::Gbm;
use crate::diffusion::ou::Ou;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;
use crate::traits::process::sample_map_chunked;

/// Scalar drift / diffusion families the device kernels know how to step.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum EulerSpec<T: FloatExt> {
  /// `dX = μX dt + σX dW`.
  GeometricBrownian { mu: T, sigma: T },
  /// `dX = θ(μ − X) dt + σ dW`.
  OrnsteinUhlenbeck { theta: T, mu: T, sigma: T },
  /// `dX = κ(θ − X) dt + σ√X dW`, stepped with full truncation (Lord,
  /// Koekkoek & van Dijk 2010): the recursion runs on an auxiliary process
  /// whose positive part enters drift, diffusion and the reported path.
  SquareRoot { kappa: T, theta: T, sigma: T },
}

impl<T: FloatExt> EulerSpec<T> {
  /// Family code and the four parameter slots the device kernels read. The
  /// layout is the kernels' ABI and stays inside the crate, so it can widen
  /// for a new family without a breaking change. Only the device kernels
  /// read it, so a build without any device feature has no caller.
  #[cfg_attr(
    not(any(
      feature = "metal",
      feature = "cuda-native",
      feature = "cubecl-cuda",
      feature = "cubecl-wgpu"
    )),
    allow(dead_code)
  )]
  pub(crate) fn encode(&self) -> (u32, [T; 4]) {
    match *self {
      EulerSpec::GeometricBrownian { mu, sigma } => (0, [mu, sigma, T::zero(), T::zero()]),
      EulerSpec::OrnsteinUhlenbeck { theta, mu, sigma } => (1, [theta, mu, sigma, T::zero()]),
      EulerSpec::SquareRoot {
        kappa,
        theta,
        sigma,
      } => (2, [kappa, theta, sigma, T::zero()]),
    }
  }
}

/// A process the device kernels can run: its coefficients, initial value,
/// grid, horizon and the seed the launch derives from the process's seed
/// source.
pub trait EulerCoefficients<T: FloatExt>: ProcessExt<T, Output = Array1<T>> {
  fn euler_spec(&self) -> EulerSpec<T>;
  fn initial_value(&self) -> T;
  /// Number of grid points including `t = 0`.
  fn grid_points(&self) -> usize;
  fn horizon(&self) -> T;
  /// One draw from the process's seed source: reproducible for
  /// `Deterministic`, fresh entropy for `Unseeded`.
  fn device_seed(&self) -> u64;
}

/// Device capability: `m` paths of `process`, each an `n`-vector whose entry
/// 0 is the initial value.
///
/// The scalar is a trait parameter and a device implements the capability
/// only for the precision its kernel computes in: `CudaNative` for `f32` and
/// `f64`, `MetalNative` and `CubeCl` for `f32` alone, the CPU devices for
/// both. `Gbm<f64>` on `MetalNative` is a compile error.
pub trait EulerBackend<T: FloatExt>: Backend {
  /// `true` for the GPU markers, whose paths come from the kernel; `false`
  /// for the CPU devices, whose paths come from the process's own sampler.
  const DEVICE: bool;

  /// Paths `first .. first + m` of the launch stream seeded by `seed`, or why
  /// the device could not produce them. The kernels hash
  /// `(first + path, step, seed)`, so a batch produced in chunks under one
  /// seed is bit-identical to one launch of the whole batch. The CPU devices
  /// ignore `first` and `seed`: their stream is sequential, and they sample
  /// `m` fresh paths.
  fn try_euler_paths_seeded<P: EulerCoefficients<T>>(
    process: &P,
    first: usize,
    m: usize,
    seed: u64,
  ) -> Result<Vec<Array1<T>>, DeviceError>;

  /// Paths `first .. first + m` under one draw of the process's seed source.
  fn try_euler_paths_from<P: EulerCoefficients<T>>(
    process: &P,
    first: usize,
    m: usize,
  ) -> Result<Vec<Array1<T>>, DeviceError> {
    Self::try_euler_paths_seeded(process, first, m, process.device_seed())
  }

  /// `m` paths, or why the device could not produce them; one seed draw,
  /// launched in chunks that fit [`crate::device::batch_budget_bytes`].
  fn try_euler_paths<P: EulerCoefficients<T>>(
    process: &P,
    m: usize,
  ) -> Result<Vec<Array1<T>>, DeviceError> {
    let seed = process.device_seed();
    let rows = crate::device::chunk_rows(process.grid_points(), std::mem::size_of::<T>());
    let mut out = Vec::with_capacity(m);
    let mut first = 0;
    while first < m {
      let len = rows.min(m - first);
      out.extend(Self::try_euler_paths_seeded(process, first, len, seed)?);
      first += len;
    }
    Ok(out)
  }

  /// `f` over `m` paths, each chunk mapped in parallel before the next is
  /// launched, so the batch never has to fit in memory at once.
  fn try_euler_paths_map<P: EulerCoefficients<T>, R: Send>(
    process: &P,
    m: usize,
    f: impl Fn(&Array1<T>) -> R + Sync,
  ) -> Result<Vec<R>, DeviceError> {
    use rayon::prelude::*;
    let seed = process.device_seed();
    let rows = crate::device::chunk_rows(process.grid_points(), std::mem::size_of::<T>());
    let mut out = Vec::with_capacity(m);
    let mut first = 0;
    while first < m {
      let len = rows.min(m - first);
      let chunk = Self::try_euler_paths_seeded(process, first, len, seed)?;
      out.extend(chunk.par_iter().map(&f).collect::<Vec<R>>());
      first += len;
    }
    Ok(out)
  }

  /// The batch as one `m × n` matrix. The device back-ends hand their launch
  /// buffer over as is, so a consumer that wants a matrix (the Python module,
  /// a column-wise estimator) skips the re-layout into rows; the default
  /// stacks [`try_euler_paths`](Self::try_euler_paths).
  fn try_euler_matrix<P: EulerCoefficients<T>>(
    process: &P,
    m: usize,
  ) -> Result<Array2<T>, DeviceError> {
    let rows = Self::try_euler_paths(process, m)?;
    let n = rows.first().map_or(process.grid_points(), |r| r.len());
    let mut out = Array2::<T>::zeros((m, n));
    for (i, row) in rows.iter().enumerate() {
      out.row_mut(i).assign(row);
    }
    Ok(out)
  }

  /// [`try_euler_paths`](Self::try_euler_paths), panicking with the device's
  /// error; [`Backend::probe`] first turns that failure into a `Result`.
  fn euler_paths<P: EulerCoefficients<T>>(process: &P, m: usize) -> Vec<Array1<T>> {
    Self::try_euler_paths(process, m).unwrap_or_else(crate::device::device_panic)
  }

  /// [`try_euler_paths_map`](Self::try_euler_paths_map), panicking with the
  /// device's error.
  fn euler_paths_map<P: EulerCoefficients<T>, R: Send>(
    process: &P,
    m: usize,
    f: impl Fn(&Array1<T>) -> R + Sync,
  ) -> Vec<R> {
    Self::try_euler_paths_map(process, m, f).unwrap_or_else(crate::device::device_panic)
  }
}

/// The CPU path is the process's own sampler.
impl<T: FloatExt> EulerBackend<T> for Cpu {
  const DEVICE: bool = false;
  fn try_euler_paths_seeded<P: EulerCoefficients<T>>(
    process: &P,
    _first: usize,
    m: usize,
    _seed: u64,
  ) -> Result<Vec<Array1<T>>, DeviceError> {
    Ok(process.sample_par(m))
  }

  /// The host never draws a device seed: its stream is the process's own.
  fn try_euler_paths_from<P: EulerCoefficients<T>>(
    process: &P,
    _first: usize,
    m: usize,
  ) -> Result<Vec<Array1<T>>, DeviceError> {
    Ok(process.sample_par(m))
  }

  /// The host stream is the process's own `sample_par`, never chunked here.
  fn try_euler_paths<P: EulerCoefficients<T>>(
    process: &P,
    m: usize,
  ) -> Result<Vec<Array1<T>>, DeviceError> {
    Ok(process.sample_par(m))
  }

  /// The host map is the process's own chunked `sample_map`.
  fn try_euler_paths_map<P: EulerCoefficients<T>, R: Send>(
    process: &P,
    m: usize,
    f: impl Fn(&Array1<T>) -> R + Sync,
  ) -> Result<Vec<R>, DeviceError> {
    Ok(sample_map_chunked(process, m, f))
  }
}

/// Accelerate is a CPU device (vDSP): the process's own sampler as well.
#[cfg(feature = "accelerate")]
impl<T: FloatExt> EulerBackend<T> for crate::device::Accelerate {
  const DEVICE: bool = false;
  fn try_euler_paths_seeded<P: EulerCoefficients<T>>(
    process: &P,
    _first: usize,
    m: usize,
    _seed: u64,
  ) -> Result<Vec<Array1<T>>, DeviceError> {
    Ok(process.sample_par(m))
  }

  /// The host never draws a device seed: its stream is the process's own.
  fn try_euler_paths_from<P: EulerCoefficients<T>>(
    process: &P,
    _first: usize,
    m: usize,
  ) -> Result<Vec<Array1<T>>, DeviceError> {
    Ok(process.sample_par(m))
  }

  /// The host stream is the process's own `sample_par`, never chunked here.
  fn try_euler_paths<P: EulerCoefficients<T>>(
    process: &P,
    m: usize,
  ) -> Result<Vec<Array1<T>>, DeviceError> {
    Ok(process.sample_par(m))
  }

  /// The host map is the process's own chunked `sample_map`.
  fn try_euler_paths_map<P: EulerCoefficients<T>, R: Send>(
    process: &P,
    m: usize,
    f: impl Fn(&Array1<T>) -> R + Sync,
  ) -> Result<Vec<R>, DeviceError> {
    Ok(sample_map_chunked(process, m, f))
  }
}

fn draw_seed<S: SeedExt>(seed: &S) -> u64 {
  rand::Rng::random(&mut seed.rng())
}

impl<T: FloatExt, S: SeedExt, B: EulerBackend<T>> EulerCoefficients<T> for Gbm<T, S, B> {
  fn euler_spec(&self) -> EulerSpec<T> {
    EulerSpec::GeometricBrownian {
      mu: self.mu,
      sigma: self.sigma,
    }
  }

  fn initial_value(&self) -> T {
    self.x0.unwrap_or(T::one())
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn device_seed(&self) -> u64 {
    draw_seed(&self.seed)
  }
}

impl<T: FloatExt, S: SeedExt, B: EulerBackend<T>> EulerCoefficients<T> for Ou<T, S, B> {
  fn euler_spec(&self) -> EulerSpec<T> {
    EulerSpec::OrnsteinUhlenbeck {
      theta: self.theta,
      mu: self.mu,
      sigma: self.sigma,
    }
  }

  fn initial_value(&self) -> T {
    self.x0.unwrap_or(T::zero())
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn device_seed(&self) -> u64 {
    draw_seed(&self.seed)
  }
}

impl<T: FloatExt, S: SeedExt, B: EulerBackend<T>> EulerCoefficients<T> for Cir<T, S, B> {
  fn euler_spec(&self) -> EulerSpec<T> {
    EulerSpec::SquareRoot {
      kappa: self.theta,
      theta: self.mu,
      sigma: self.sigma,
    }
  }

  fn initial_value(&self) -> T {
    self.x0.unwrap_or(T::zero())
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn device_seed(&self) -> u64 {
    draw_seed(&self.seed)
  }
}

macro_rules! try_sample_par {
  ($ty:ident) => {
    impl<T: FloatExt, S: SeedExt, B: EulerBackend<T>> $ty<T, S, B> {
      /// `m` paths, or the device's error instead of the panic
      /// [`ProcessExt::sample_par`] raises when the device cannot serve the
      /// request. On the CPU devices this is always `Ok` and bit-identical
      /// to `sample_par`.
      pub fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, DeviceError> {
        B::try_euler_paths(self, m)
      }

      /// The batch as one `m × n` matrix: on a device back-end the launch
      /// buffer itself, without a re-layout into rows.
      pub fn try_sample_matrix(&self, m: usize) -> Result<Array2<T>, DeviceError> {
        B::try_euler_matrix(self, m)
      }
    }
  };
}

try_sample_par!(Gbm);
try_sample_par!(Ou);
try_sample_par!(Cir);

#[cfg(feature = "cuda-native")]
pub mod cuda_native;
#[cfg(any(feature = "cubecl-cuda", feature = "cubecl-wgpu"))]
pub mod gpu;
#[cfg(feature = "metal")]
pub mod metal;

#[cfg(test)]
mod tests;

/// A single-precision device refuses an `f64` process at compile time.
///
/// ```compile_fail,E0277
/// use stochastic_rs_core::simd_rng::Unseeded;
/// use stochastic_rs_stochastic::device::MetalNative;
/// use stochastic_rs_stochastic::diffusion::gbm::Gbm;
/// use stochastic_rs_stochastic::traits::ProcessExt;
///
/// let gbm = Gbm::<f64, _>::new(0.05, 0.2, 16, None, None, Unseeded);
/// let _ = gbm.on::<MetalNative>().sample();
/// ```
#[cfg(feature = "metal")]
pub mod precision_guard {}

#[cfg(feature = "python")]
pub mod python {
  //! Python surface of the device layer: probing a device and choosing the
  //! ordinal. Sampling on a device goes through the process classes'
  //! `device=` argument.

  use pyo3::exceptions::PyValueError;
  use pyo3::prelude::*;

  use crate::device::Cpu;
  use crate::device::DeviceError;

  /// Runs `m` paths of the process on the requested device through
  /// `.on::<B>()`; every arm is the same call on a different marker.
  /// A device failure is a Python `RuntimeError` carrying the device's message.
  fn device_err(e: DeviceError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
  }

  /// Chooses the device ordinal the device back-ends open (CUDA ordinal,
  /// Metal device index, CubeCL device index); process-wide, `0` by default
  /// or `STOCHASTIC_RS_DEVICE`. See `probe_device` to check what it opens.
  #[pyfunction]
  pub fn select_device(ordinal: usize) {
    crate::device::select_device(ordinal);
  }

  /// Opens the named device (`"cpu"`, `"gpu"`, `"cuda-native"`, `"metal"`,
  /// `"cubecl"`, `"accelerate"`) and describes it as a dict with `backend`,
  /// `name`, `precisions` and `ordinal`; raises `RuntimeError` with the
  /// device's own message when it cannot be used, `ValueError` for a device
  /// this build does not carry.
  #[pyfunction]
  pub fn probe_device<'py>(
    py: Python<'py>,
    device: &str,
  ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use crate::device::Backend;
    use crate::device::DeviceInfo;
    fn describe<'py>(
      py: Python<'py>,
      info: DeviceInfo,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
      let d = pyo3::types::PyDict::new(py);
      d.set_item("backend", info.backend)?;
      d.set_item("name", info.name)?;
      d.set_item("precisions", info.precisions.to_vec())?;
      d.set_item("ordinal", info.ordinal)?;
      Ok(d)
    }
    let missing = |what: &str, feature: &str| {
      PyValueError::new_err(format!(
        "this build has no {what}; rebuild with the {feature} feature"
      ))
    };
    let info = match device.to_ascii_lowercase().as_str() {
      "cpu" => Cpu::probe(),
      "accelerate" => {
        #[cfg(feature = "accelerate")]
        {
          crate::device::Accelerate::probe()
        }

        #[cfg(not(feature = "accelerate"))]
        {
          return Err(missing("Accelerate back-end", "accelerate"));
        }
      }
      "cuda-native" | "cuda_native" => {
        #[cfg(feature = "cuda-native")]
        {
          crate::device::CudaNative::probe()
        }

        #[cfg(not(feature = "cuda-native"))]
        {
          return Err(missing("native CUDA runtime", "cuda-native"));
        }
      }
      "metal" => {
        #[cfg(feature = "metal")]
        {
          crate::device::MetalNative::probe()
        }

        #[cfg(not(feature = "metal"))]
        {
          return Err(missing("native Metal runtime", "metal"));
        }
      }
      "cubecl" => {
        #[cfg(any(feature = "cubecl-cuda", feature = "cubecl-wgpu"))]
        {
          crate::device::CubeCl::probe()
        }

        #[cfg(not(any(feature = "cubecl-cuda", feature = "cubecl-wgpu")))]
        {
          return Err(missing("CubeCL runtime", "cubecl-cuda or cubecl-wgpu"));
        }
      }
      "gpu" => {
        #[cfg(feature = "cuda-native")]
        {
          crate::device::CudaNative::probe()
        }

        #[cfg(all(feature = "metal", not(feature = "cuda-native")))]
        {
          crate::device::MetalNative::probe()
        }

        #[cfg(all(
          any(feature = "cubecl-cuda", feature = "cubecl-wgpu"),
          not(feature = "metal"),
          not(feature = "cuda-native")
        ))]
        {
          crate::device::CubeCl::probe()
        }

        #[cfg(not(any(
          feature = "cuda-native",
          feature = "metal",
          feature = "cubecl-cuda",
          feature = "cubecl-wgpu"
        )))]
        {
          return Err(missing(
            "GPU runtime",
            "cuda-native, metal, cubecl-cuda or cubecl-wgpu",
          ));
        }
      }
      other => {
        return Err(PyValueError::new_err(format!(
          "unknown device {other:?}; use cpu, gpu, cuda-native, metal, cubecl or accelerate"
        )));
      }
    };
    describe(py, info.map_err(device_err)?)
  }
}
