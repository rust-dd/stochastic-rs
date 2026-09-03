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
use stochastic_rs_core::simd_rng::SeedExt;

use crate::device::Backend;
use crate::device::Cpu;
use crate::device::DeviceError;
use crate::diffusion::cir::Cir;
use crate::diffusion::gbm::Gbm;
use crate::diffusion::ou::Ou;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

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
      feature = "gpu-cuda",
      feature = "gpu-wgpu"
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
  /// `m` paths, or why the device could not produce them.
  fn try_euler_paths<P: EulerCoefficients<T>>(
    process: &P,
    m: usize,
  ) -> Result<Vec<Array1<T>>, DeviceError>;

  /// [`try_euler_paths`](Self::try_euler_paths), panicking with the device's
  /// error; [`Backend::probe`] first turns that failure into a `Result`.
  fn euler_paths<P: EulerCoefficients<T>>(process: &P, m: usize) -> Vec<Array1<T>> {
    Self::try_euler_paths(process, m).unwrap_or_else(crate::device::device_panic)
  }
}

/// The CPU path is the process's own sampler.
impl<T: FloatExt> EulerBackend<T> for Cpu {
  const DEVICE: bool = false;
  fn try_euler_paths<P: EulerCoefficients<T>>(
    process: &P,
    m: usize,
  ) -> Result<Vec<Array1<T>>, DeviceError> {
    Ok(process.sample_par(m))
  }
}

/// Accelerate is a CPU device (vDSP): the process's own sampler as well.
#[cfg(feature = "accelerate")]
impl<T: FloatExt> EulerBackend<T> for crate::device::Accelerate {
  const DEVICE: bool = false;
  fn try_euler_paths<P: EulerCoefficients<T>>(
    process: &P,
    m: usize,
  ) -> Result<Vec<Array1<T>>, DeviceError> {
    Ok(process.sample_par(m))
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
    }
  };
}

try_sample_par!(Gbm);
try_sample_par!(Ou);
try_sample_par!(Cir);

#[cfg(feature = "cuda-native")]
pub mod cuda_native;
#[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu"))]
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
  //! Python surface of the Euler engine: one function over the scalar
  //! families, `device="cpu"` always, the device names when their back-end
  //! is compiled in; `float32` arrays from the single-precision devices.

  use numpy::IntoPyArray;
  use pyo3::IntoPyObjectExt;
  use pyo3::exceptions::PyValueError;
  use pyo3::prelude::*;
  use stochastic_rs_core::simd_rng::Deterministic;

  use crate::device::Cpu;
  use crate::device::DeviceError;
  use crate::diffusion::cir::Cir;
  use crate::diffusion::gbm::Gbm;
  use crate::diffusion::ou::Ou;
  use crate::traits::FloatExt;

  /// Runs `m` paths of the process on the requested device through
  /// `.on::<B>()`; every arm is the same call on a different marker.
  /// A device failure is a Python `RuntimeError` carrying the device's message.
  fn device_err(e: DeviceError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
  }

  /// Which scalar a device computes in: the CPU and native CUDA paths keep
  /// `f64`, the Metal and CubeCL kernels are single precision and the array
  /// they return says so.
  enum Rows {
    F64(Vec<ndarray::Array1<f64>>),
    #[cfg(any(feature = "metal", feature = "gpu-cuda", feature = "gpu-wgpu"))]
    F32(Vec<ndarray::Array1<f32>>),
  }

  fn stack<'py, T: FloatExt + numpy::Element>(
    py: Python<'py>,
    rows: Vec<ndarray::Array1<T>>,
    n: usize,
  ) -> PyResult<Py<PyAny>> {
    let mut paths = ndarray::Array2::<T>::zeros((rows.len(), n));
    for (i, row) in rows.iter().enumerate() {
      paths.row_mut(i).assign(row);
    }
    paths.into_pyarray(py).into_py_any(py)
  }

  fn gbm<T: FloatExt>(p: &[f64], x0: f64, n: usize, t: f64, seed: u64) -> Gbm<T, Deterministic> {
    Gbm::new(
      T::from_f64_fast(p[0]),
      T::from_f64_fast(p[1]),
      n,
      Some(T::from_f64_fast(x0)),
      Some(T::from_f64_fast(t)),
      Deterministic::new(seed),
    )
  }

  fn ou<T: FloatExt>(p: &[f64], x0: f64, n: usize, t: f64, seed: u64) -> Ou<T, Deterministic> {
    Ou::new(
      T::from_f64_fast(p[0]),
      T::from_f64_fast(p[1]),
      T::from_f64_fast(p[2]),
      n,
      Some(T::from_f64_fast(x0)),
      Some(T::from_f64_fast(t)),
      Deterministic::new(seed),
    )
  }

  fn cir<T: FloatExt>(p: &[f64], x0: f64, n: usize, t: f64, seed: u64) -> Cir<T, Deterministic> {
    Cir::new(
      T::from_f64_fast(p[0]),
      T::from_f64_fast(p[1]),
      T::from_f64_fast(p[2]),
      n,
      Some(T::from_f64_fast(x0)),
      Some(T::from_f64_fast(t)),
      None,
      Deterministic::new(seed),
    )
  }

  /// `$p64` / `$p32` build the process in the scalar the chosen device
  /// computes in; only the arm that runs is evaluated.
  macro_rules! on_device {
    ($p64:expr, $p32:expr, $m:expr, $py:expr, $device:expr) => {
      match $device {
        "cpu" => Rows::F64($py.detach(|| $p64.on::<Cpu>().try_sample_par($m)).map_err(device_err)?),
        "cuda-native" | "cuda_native" => {
          #[cfg(feature = "cuda-native")]
          {
            Rows::F64($py.detach(|| $p64.on::<crate::device::CudaNative>().try_sample_par($m)).map_err(device_err)?)
          }

          #[cfg(not(feature = "cuda-native"))]
          {
            return Err(PyValueError::new_err(
              "this build has no native CUDA runtime; rebuild with the cuda-native feature",
            ));
          }
        }
        "metal" => {
          #[cfg(feature = "metal")]
          {
            Rows::F32($py.detach(|| $p32.on::<crate::device::MetalNative>().try_sample_par($m)).map_err(device_err)?)
          }

          #[cfg(not(feature = "metal"))]
          {
            return Err(PyValueError::new_err(
              "this build has no native Metal runtime; rebuild with the metal feature",
            ));
          }
        }
        "cubecl" => {
          #[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu"))]
          {
            Rows::F32($py.detach(|| $p32.on::<crate::device::CubeCl>().try_sample_par($m)).map_err(device_err)?)
          }

          #[cfg(not(any(feature = "gpu-cuda", feature = "gpu-wgpu")))]
          {
            return Err(PyValueError::new_err(
              "this build has no CubeCL runtime; rebuild with the gpu-cuda or gpu-wgpu feature",
            ));
          }
        }
        // The first compiled device back-end: native CUDA, then native Metal, then CubeCL.
        "gpu" => {
          #[cfg(feature = "cuda-native")]
          {
            Rows::F64($py.detach(|| $p64.on::<crate::device::CudaNative>().try_sample_par($m)).map_err(device_err)?)
          }

          #[cfg(all(feature = "metal", not(feature = "cuda-native")))]
          {
            Rows::F32($py.detach(|| $p32.on::<crate::device::MetalNative>().try_sample_par($m)).map_err(device_err)?)
          }

          #[cfg(all(
            any(feature = "gpu-cuda", feature = "gpu-wgpu"),
            not(feature = "metal"),
            not(feature = "cuda-native")
          ))]
          {
            Rows::F32($py.detach(|| $p32.on::<crate::device::CubeCl>().try_sample_par($m)).map_err(device_err)?)
          }

          #[cfg(not(any(
            feature = "cuda-native",
            feature = "metal",
            feature = "gpu-cuda",
            feature = "gpu-wgpu"
          )))]
          {
            return Err(PyValueError::new_err(
              "this build has no GPU runtime; rebuild with the cuda-native, metal, gpu-cuda or gpu-wgpu feature",
            ));
          }
        }
        other => {
          return Err(PyValueError::new_err(format!(
            "unknown device {other:?}; use cpu, gpu, cuda-native, metal or cubecl"
          )));
        }
      }
    };
  }

  /// `m` Euler paths of a scalar family on a device, as an `(m, n)` array.
  ///
  /// `family` is `"gbm"` (`[mu, sigma]`), `"ou"` (`[theta, mu, sigma]`) or
  /// `"cir"` (`[kappa, theta, sigma]`); `device` is `"cpu"`, `"gpu"` (the
  /// first compiled device back-end), or one of `"cuda-native"`, `"metal"`,
  /// `"cubecl"` (each needs the matching cargo feature of the build). The
  /// array is `float64` from the CPU and native CUDA paths and `float32` from
  /// Metal and CubeCL, whose kernels compute in single precision.
  #[pyfunction]
  #[pyo3(signature = (family, params, x0, n, t, m, seed=42, device="cpu"))]
  #[allow(clippy::too_many_arguments)]
  pub fn euler_paths<'py>(
    py: Python<'py>,
    family: &str,
    params: Vec<f64>,
    x0: f64,
    n: usize,
    t: f64,
    m: usize,
    seed: u64,
    device: &str,
  ) -> PyResult<Py<PyAny>> {
    let need = |k: usize| {
      if params.len() != k {
        Err(PyValueError::new_err(format!(
          "{family} takes {k} parameters, got {}",
          params.len()
        )))
      } else {
        Ok(())
      }
    };
    let device = device.to_ascii_lowercase();
    let p = params.as_slice();
    let rows = match family.to_ascii_lowercase().as_str() {
      "gbm" => {
        need(2)?;
        on_device!(
          gbm::<f64>(p, x0, n, t, seed),
          gbm::<f32>(p, x0, n, t, seed),
          m,
          py,
          device.as_str()
        )
      }
      "ou" => {
        need(3)?;
        on_device!(
          ou::<f64>(p, x0, n, t, seed),
          ou::<f32>(p, x0, n, t, seed),
          m,
          py,
          device.as_str()
        )
      }
      "cir" => {
        need(3)?;
        on_device!(
          cir::<f64>(p, x0, n, t, seed),
          cir::<f32>(p, x0, n, t, seed),
          m,
          py,
          device.as_str()
        )
      }
      other => {
        return Err(PyValueError::new_err(format!(
          "unknown Euler family {other:?}; use gbm, ou or cir"
        )));
      }
    };
    match rows {
      Rows::F64(rows) => stack(py, rows, n),
      #[cfg(any(feature = "metal", feature = "gpu-cuda", feature = "gpu-wgpu"))]
      Rows::F32(rows) => stack(py, rows, n),
    }
  }
}
