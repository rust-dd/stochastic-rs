//! # Euler engine
//!
//! $$
//! X_{i+1} = X_i + b(X_i)\,\Delta t + \sigma(X_i)\sqrt{\Delta t}\,Z_i,\qquad
//! Z_i \sim \mathcal N(0, 1)\ \text{i.i.d.}
//! $$
//!
//! Device-side path generation for the diffusions whose coefficients are a
//! handful of scalars. One entry point, [`sample_paths`], parameterised by a
//! [`Backend`] with the [`EulerBackend`] capability:
//!
//! - [`Cpu`] (and `Accelerate`, a CPU device) is **the process's own
//!   sampler** — `sample_par` with the seed pinned through `Deterministic` —
//!   so nothing is re-implemented on the host: GBM keeps its exact log-normal
//!   scheme, OU and CIR their SIMD Euler steppers.
//! - The GPU back-ends run one device thread per path with the whole
//!   Euler–Maruyama recursion in the kernel and Box–Muller normals from a
//!   counter hash of `(path, step, seed)`: `CubeCl` (features `gpu-cuda` /
//!   `gpu-wgpu`: CUDA, Metal, Vulkan or WebGPU through CubeCL, `f32`),
//!   `CudaNative` (feature `cuda-native`: cudarc + NVRTC, `f32` or `f64` after
//!   `T`) and `MetalNative` (feature `metal`: hand-written MSL, `f32`).
//!
//! A process joins the engine by describing its coefficients as an
//! [`EulerSpec`] through [`EulerCoefficients`] — [`Gbm`], [`Ou`] and [`Cir`]
//! do. The device kernels share one integer hash, so the device back-ends
//! agree with each other seed for seed up to libm rounding; the CPU path is
//! the process's own stream, so CPU and device paths agree in distribution,
//! not bit for bit.
//!
//! References: Kloeden, P. E. & Platen, E. (1992), *Numerical Solution of
//! Stochastic Differential Equations*, Springer, §10.2 (Euler–Maruyama);
//! Lord, R., Koekkoek, R. & van Dijk, D. (2010), *A comparison of biased
//! simulation schemes for stochastic volatility models*, Quantitative Finance
//! 10(2), 177–194 (full truncation, used by the device kernels for CIR).

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;

use crate::device::Backend;
use crate::device::Cpu;
use crate::diffusion::cir::Cir;
use crate::diffusion::gbm::Gbm;
use crate::diffusion::ou::Ou;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Scalar drift / diffusion families the device kernels know how to step.
#[derive(Clone, Copy, Debug, PartialEq)]
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
  /// Family code and the four parameter slots the device kernels read.
  pub fn encode(&self) -> (u32, [T; 4]) {
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

/// A process the Euler engine can run on a device: its coefficients, initial
/// value, grid and horizon for the kernels, and a deterministically re-seeded
/// copy of itself for the CPU path.
pub trait EulerCoefficients<T: FloatExt>: ProcessExt<T, Output = Array1<T>> {
  /// The same process with its seed replaced by `Deterministic::new(seed)`.
  type Seeded: ProcessExt<T, Output = Array1<T>>;
  fn seeded(&self, seed: u64) -> Self::Seeded;
  fn euler_spec(&self) -> EulerSpec<T>;
  fn initial_value(&self) -> T;
  /// Number of grid points including `t = 0`.
  fn grid_points(&self) -> usize;
  fn horizon(&self) -> T;
}

/// Device capability: `m` paths of `process` on the device, as an `m × n`
/// matrix whose column 0 holds the initial value.
pub trait EulerBackend: Backend {
  fn euler_paths<T: FloatExt, P: EulerCoefficients<T>>(
    process: &P,
    m: usize,
    seed: u64,
  ) -> Array2<T>;
}

/// Runs `m` paths of `process` on backend `B` with the seed `seed`.
pub fn sample_paths<T: FloatExt, B: EulerBackend, P: EulerCoefficients<T>>(
  process: &P,
  m: usize,
  seed: u64,
) -> Array2<T> {
  B::euler_paths(process, m, seed)
}

/// Stacks `sample_par` output into the engine's `m × n` layout.
fn stack_rows<T: FloatExt>(rows: Vec<Array1<T>>, n: usize) -> Array2<T> {
  let m = rows.len();
  let mut out = Array2::<T>::zeros((m, n));
  for (i, row) in rows.iter().enumerate() {
    out.row_mut(i).assign(row);
  }
  out
}

/// The CPU path is the process's own sampler.
impl EulerBackend for Cpu {
  fn euler_paths<T: FloatExt, P: EulerCoefficients<T>>(
    process: &P,
    m: usize,
    seed: u64,
  ) -> Array2<T> {
    if m == 0 {
      return Array2::<T>::zeros((0, process.grid_points()));
    }
    stack_rows(process.seeded(seed).sample_par(m), process.grid_points())
  }
}

/// Accelerate is a CPU device (vDSP): the process's own sampler as well.
#[cfg(feature = "accelerate")]
impl EulerBackend for crate::device::Accelerate {
  fn euler_paths<T: FloatExt, P: EulerCoefficients<T>>(
    process: &P,
    m: usize,
    seed: u64,
  ) -> Array2<T> {
    Cpu::euler_paths(process, m, seed)
  }
}

impl<T: FloatExt, S: SeedExt> EulerCoefficients<T> for Gbm<T, S> {
  type Seeded = Gbm<T, Deterministic>;
  fn seeded(&self, seed: u64) -> Self::Seeded {
    Gbm::new(
      self.mu,
      self.sigma,
      self.n,
      self.x0,
      self.t,
      Deterministic::new(seed),
    )
  }
  fn euler_spec(&self) -> EulerSpec<T> {
    EulerSpec::GeometricBrownian {
      mu: self.mu,
      sigma: self.sigma,
    }
  }
  fn initial_value(&self) -> T {
    self.x0.unwrap_or(T::from_usize_(100))
  }
  fn grid_points(&self) -> usize {
    self.n
  }
  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }
}

impl<T: FloatExt, S: SeedExt> EulerCoefficients<T> for Ou<T, S> {
  type Seeded = Ou<T, Deterministic>;
  fn seeded(&self, seed: u64) -> Self::Seeded {
    Ou::new(
      self.theta,
      self.mu,
      self.sigma,
      self.n,
      self.x0,
      self.t,
      Deterministic::new(seed),
    )
  }
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
}

impl<T: FloatExt, S: SeedExt> EulerCoefficients<T> for Cir<T, S> {
  type Seeded = Cir<T, Deterministic>;
  fn seeded(&self, seed: u64) -> Self::Seeded {
    Cir::new(
      self.theta,
      self.mu,
      self.sigma,
      self.n,
      self.x0,
      self.t,
      self.use_sym,
      Deterministic::new(seed),
    )
  }
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
}

#[cfg(feature = "cuda-native")]
pub mod cuda_native;
#[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu"))]
pub mod gpu;
#[cfg(feature = "metal")]
pub mod metal;

#[cfg(test)]
mod tests;

#[cfg(feature = "python")]
pub mod python {
  //! Python surface of the Euler engine: one function over the scalar
  //! families, `device="cpu"` always, `device="gpu"` when a CubeCL runtime is
  //! compiled in.

  use numpy::IntoPyArray;
  use pyo3::exceptions::PyValueError;
  use pyo3::prelude::*;
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::EulerBackend;
  use super::EulerCoefficients;
  use crate::device::Cpu;
  use crate::diffusion::cir::Cir;
  use crate::diffusion::gbm::Gbm;
  use crate::diffusion::ou::Ou;

  /// Runs `m` paths of the family on the requested device.
  fn dispatch<P: EulerCoefficients<f64>>(
    py: Python<'_>,
    process: &P,
    m: usize,
    seed: u64,
    device: &str,
  ) -> PyResult<ndarray::Array2<f64>> {
    Ok(match device.to_ascii_lowercase().as_str() {
      "cpu" => py.detach(|| Cpu::euler_paths(process, m, seed)),
      "cuda-native" | "cuda_native" => {
        #[cfg(feature = "cuda-native")]
        {
          py.detach(|| crate::device::CudaNative::euler_paths(process, m, seed))
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
          py.detach(|| crate::device::MetalNative::euler_paths(process, m, seed))
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
          py.detach(|| crate::device::CubeCl::euler_paths(process, m, seed))
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
          py.detach(|| crate::device::CudaNative::euler_paths(process, m, seed))
        }
        #[cfg(all(feature = "metal", not(feature = "cuda-native")))]
        {
          py.detach(|| crate::device::MetalNative::euler_paths(process, m, seed))
        }
        #[cfg(all(
          any(feature = "gpu-cuda", feature = "gpu-wgpu"),
          not(feature = "metal"),
          not(feature = "cuda-native")
        ))]
        {
          py.detach(|| crate::device::CubeCl::euler_paths(process, m, seed))
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
    })
  }

  /// `m × n` paths of a scalar diffusion: the process's own sampler on the
  /// CPU, the Euler–Maruyama kernel on a device. `family` is `"gbm"`
  /// (`[mu, sigma]`), `"ou"` (`[theta, mu, sigma]`) or `"cir"`
  /// (`[kappa, theta, sigma]`); `device` is `"cpu"`, `"gpu"` (the first
  /// compiled device back-end), or one of `"cuda-native"`, `"metal"`,
  /// `"cubecl"` (each needs the matching cargo feature of the build).
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
  ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
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
    let paths = match family.to_ascii_lowercase().as_str() {
      "gbm" => {
        need(2)?;
        let process = Gbm::new(params[0], params[1], n, Some(x0), Some(t), Unseeded);
        dispatch(py, &process, m, seed, device)?
      }
      "ou" => {
        need(3)?;
        let process = Ou::new(
          params[0],
          params[1],
          params[2],
          n,
          Some(x0),
          Some(t),
          Unseeded,
        );
        dispatch(py, &process, m, seed, device)?
      }
      "cir" => {
        need(3)?;
        let process = Cir::new(
          params[0],
          params[1],
          params[2],
          n,
          Some(x0),
          Some(t),
          None,
          Unseeded,
        );
        dispatch(py, &process, m, seed, device)?
      }
      other => {
        return Err(PyValueError::new_err(format!(
          "unknown Euler family {other:?}; use gbm, ou or cir"
        )));
      }
    };
    Ok(paths.into_pyarray(py))
  }
}
