//! # Euler engine
//!
//! $$
//! X_{i+1} = X_i + b(X_i)\,\Delta t + \sigma(X_i)\sqrt{\Delta t}\,Z_i,\qquad
//! Z_i \sim \mathcal N(0, 1)\ \text{i.i.d.}
//! $$
//!
//! One Euler–Maruyama stepper for the diffusions whose coefficients are a
//! handful of scalars, on any [`Backend`] with the [`EulerBackend`]
//! capability. [`Cpu`] (and `Accelerate`, a CPU device) runs the recursion
//! with the crate's SIMD normal generator, one rayon task per path. The GPU
//! back-ends run one device thread per path with the whole recursion in the
//! kernel and Box–Muller normals from a counter hash of `(path, step, seed)`:
//! `CubeCl` (features `gpu-cuda` / `gpu-wgpu`: CUDA, Metal, Vulkan or WebGPU
//! through CubeCL, `f32`), `CudaNative` (feature `cuda-native`: cudarc +
//! NVRTC, `f32` or `f64` after `T`) and `MetalNative` (feature `metal`:
//! hand-written MSL, `f32`). A process joins the engine by describing its
//! coefficients as an [`EulerSpec`] through [`EulerCoefficients`]. The
//! stepper is the same formula everywhere and the device kernels share one
//! integer hash, so the device back-ends agree with each other seed for seed
//! up to libm rounding, while the CPU draws its own stream: CPU and device
//! paths agree in distribution, not bit for bit.
//!
//! References: Kloeden, P. E. & Platen, E. (1992), *Numerical Solution of
//! Stochastic Differential Equations*, Springer, §10.2 (Euler–Maruyama);
//! Lord, R., Koekkoek, R. & van Dijk, D. (2010), *A comparison of biased
//! simulation schemes for stochastic volatility models*, Quantitative Finance
//! 10(2), 177–194 (full truncation).

use ndarray::Array2;
use ndarray::parallel::prelude::*;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::device::Backend;
use crate::device::Cpu;
use crate::traits::FloatExt;

/// Scalar drift / diffusion families the engine knows how to step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EulerSpec<T: FloatExt> {
  /// `dX = μX dt + σX dW`.
  GeometricBrownian { mu: T, sigma: T },
  /// `dX = θ(μ − X) dt + σ dW`.
  OrnsteinUhlenbeck { theta: T, mu: T, sigma: T },
  /// `dX = κ(θ − X) dt + σ√X dW` with the full-truncation scheme of Lord,
  /// Koekkoek & van Dijk (2010): the recursion runs on an auxiliary process
  /// whose positive part enters drift, diffusion and the reported path.
  SquareRoot { kappa: T, theta: T, sigma: T },
}

impl<T: FloatExt> EulerSpec<T> {
  /// Family code and the four parameter slots the device kernel reads.
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

  /// The value a path reports for the state `x`: the state itself, except for
  /// the square-root family, whose auxiliary process is reported through its
  /// positive part.
  #[inline]
  pub fn observed(&self, x: T) -> T {
    match self {
      EulerSpec::SquareRoot { .. } => x.max(T::zero()),
      _ => x,
    }
  }

  /// One Euler step of the (auxiliary) state `x` with the standard normal `z`.
  #[inline]
  pub fn step(&self, x: T, dt: T, sqrt_dt: T, z: T) -> T {
    match *self {
      EulerSpec::GeometricBrownian { mu, sigma } => x + mu * x * dt + sigma * x * sqrt_dt * z,
      EulerSpec::OrnsteinUhlenbeck { theta, mu, sigma } => {
        x + theta * (mu - x) * dt + sigma * sqrt_dt * z
      }
      EulerSpec::SquareRoot {
        kappa,
        theta,
        sigma,
      } => {
        let positive = x.max(T::zero());
        x + kappa * (theta - positive) * dt + sigma * positive.sqrt() * sqrt_dt * z
      }
    }
  }
}

/// A process the Euler engine can run: its coefficients, initial value,
/// grid size and horizon.
pub trait EulerCoefficients<T: FloatExt> {
  fn euler_spec(&self) -> EulerSpec<T>;
  fn initial_value(&self) -> T;
  /// Number of grid points including `t = 0`.
  fn grid_points(&self) -> usize;
  fn horizon(&self) -> T;
}

/// Device capability: run `m` Euler paths of `spec` on the device.
pub trait EulerBackend: Backend {
  /// `m × n` matrix of paths; column 0 holds the initial value.
  fn euler_paths<T: FloatExt>(
    spec: EulerSpec<T>,
    x0: T,
    n: usize,
    t: T,
    m: usize,
    seed: u64,
  ) -> Array2<T>;
}

/// Runs `m` paths of `process` on backend `B`.
pub fn sample_paths<T: FloatExt, B: EulerBackend, P: EulerCoefficients<T>>(
  process: &P,
  m: usize,
  seed: u64,
) -> Array2<T> {
  B::euler_paths(
    process.euler_spec(),
    process.initial_value(),
    process.grid_points(),
    process.horizon(),
    m,
    seed,
  )
}

impl EulerBackend for Cpu {
  fn euler_paths<T: FloatExt>(
    spec: EulerSpec<T>,
    x0: T,
    n: usize,
    t: T,
    m: usize,
    seed: u64,
  ) -> Array2<T> {
    let mut out = Array2::<T>::zeros((m, n));
    if n == 0 {
      return out;
    }
    let dt = t / T::from_usize_(n.max(2) - 1);
    let sqrt_dt = dt.sqrt();
    out
      .axis_iter_mut(ndarray::Axis(0))
      .into_par_iter()
      .enumerate()
      .for_each(|(path, mut row)| {
        // One decorrelated stream per path: the seed is mixed with the path index.
        let stream =
          Deterministic::new(seed ^ (path as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let normal = SimdNormal::<T>::new(T::zero(), T::one(), &stream);
        let mut z = vec![T::zero(); n - 1];
        normal.fill_slice(&mut z);
        let mut x = x0;
        row[0] = spec.observed(x);
        for (i, &zi) in z.iter().enumerate() {
          x = spec.step(x, dt, sqrt_dt, zi);
          row[i + 1] = spec.observed(x);
        }
      });
    out
  }
}

impl<T: FloatExt, S: SeedExt> EulerCoefficients<T> for crate::diffusion::gbm::Gbm<T, S> {
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

impl<T: FloatExt, S: SeedExt> EulerCoefficients<T> for crate::diffusion::ou::Ou<T, S> {
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

impl<T: FloatExt, S: SeedExt> EulerCoefficients<T> for crate::diffusion::cir::Cir<T, S> {
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

/// Accelerate is a CPU device (vDSP), so its Euler paths are the CPU engine's.
#[cfg(feature = "accelerate")]
impl EulerBackend for crate::device::Accelerate {
  fn euler_paths<T: FloatExt>(
    spec: EulerSpec<T>,
    x0: T,
    n: usize,
    t: T,
    m: usize,
    seed: u64,
  ) -> Array2<T> {
    Cpu::euler_paths(spec, x0, n, t, m, seed)
  }
}

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

  use super::EulerBackend;
  use super::EulerSpec;
  use crate::device::Cpu;

  fn spec_from(family: &str, params: &[f64]) -> PyResult<EulerSpec<f64>> {
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
    Ok(match family.to_ascii_lowercase().as_str() {
      "gbm" => {
        need(2)?;
        EulerSpec::GeometricBrownian {
          mu: params[0],
          sigma: params[1],
        }
      }
      "ou" => {
        need(3)?;
        EulerSpec::OrnsteinUhlenbeck {
          theta: params[0],
          mu: params[1],
          sigma: params[2],
        }
      }
      "cir" => {
        need(3)?;
        EulerSpec::SquareRoot {
          kappa: params[0],
          theta: params[1],
          sigma: params[2],
        }
      }
      other => {
        return Err(PyValueError::new_err(format!(
          "unknown Euler family {other:?}; use gbm, ou or cir"
        )));
      }
    })
  }

  /// `m × n` Euler–Maruyama paths of a scalar diffusion. `family` is `"gbm"`
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
    let spec = spec_from(family, &params)?;
    let paths = match device.to_ascii_lowercase().as_str() {
      "cpu" => py.detach(|| Cpu::euler_paths(spec, x0, n, t, m, seed)),
      "cuda-native" | "cuda_native" => {
        #[cfg(feature = "cuda-native")]
        {
          py.detach(|| crate::device::CudaNative::euler_paths(spec, x0, n, t, m, seed))
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
          py.detach(|| crate::device::MetalNative::euler_paths(spec, x0, n, t, m, seed))
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
          py.detach(|| crate::device::CubeCl::euler_paths(spec, x0, n, t, m, seed))
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
          py.detach(|| crate::device::CudaNative::euler_paths(spec, x0, n, t, m, seed))
        }
        #[cfg(all(feature = "metal", not(feature = "cuda-native")))]
        {
          py.detach(|| crate::device::MetalNative::euler_paths(spec, x0, n, t, m, seed))
        }
        #[cfg(all(
          any(feature = "gpu-cuda", feature = "gpu-wgpu"),
          not(feature = "metal"),
          not(feature = "cuda-native")
        ))]
        {
          py.detach(|| crate::device::CubeCl::euler_paths(spec, x0, n, t, m, seed))
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
    };
    Ok(paths.into_pyarray(py))
  }
}
