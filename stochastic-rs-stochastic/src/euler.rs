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
//! with `.on::<B>()` and then sampled through [`ProcessExt`] as usual —
//! `gbm.on::<Metal>().sample_par(m)`.
//!
//! - [`Cpu`] (and `Accelerate`, a CPU device) is **the process's own
//!   sampler**, so nothing is re-implemented on the host: GBM keeps its exact
//!   log-normal scheme, OU and CIR their SIMD Euler steppers.
//! - The GPU back-ends run one device thread per path with the whole
//!   Euler–Maruyama recursion in the kernel and Box–Muller normals from a
//!   counter hash of `(path, step, seed)`: `Cubecl` (its CUDA runtime, or
//!   Metal / Vulkan / WebGPU through wgpu, `f32`),
//!   `Cuda` (feature `cuda`: cudarc + NVRTC, `f32` or `f64` after
//!   `T`) and `Metal` (feature `metal`: hand-written MSL, `f32`).
//!   `sample_par` is one launch for all `m` paths; `sample` launches one path.
//!
//! The device seed is drawn from the process's own seed source, so the same
//! `Deterministic` seed value gives the same device paths, consecutive calls
//! advance the stream and an `Unseeded` process draws fresh entropy, exactly
//! as on the host. The device
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
use crate::traits::PathSampler;
use crate::traits::ProcessExt;
use crate::traits::process::sample_map_chunked;
use crate::traits::process::sample_par_chunked;

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
      feature = "cuda",
      feature = "cubecl-cuda",
      feature = "cubecl-wgpu"
    )),
    allow(dead_code)
  )]
  pub(crate) fn encode(&self) -> (u32, [T; 4]) {
    use families::Family;
    match *self {
      EulerSpec::GeometricBrownian { mu, sigma } => (
        Family::GeometricBrownian.code(),
        [mu, sigma, T::zero(), T::zero()],
      ),
      EulerSpec::OrnsteinUhlenbeck { theta, mu, sigma } => (
        Family::OrnsteinUhlenbeck.code(),
        [theta, mu, sigma, T::zero()],
      ),
      EulerSpec::SquareRoot {
        kappa,
        theta,
        sigma,
      } => (Family::SquareRoot.code(), [kappa, theta, sigma, T::zero()]),
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

  /// One path from the process's own sampler, the host stream.
  fn host_sample(&self) -> Array1<T>;
}

/// The device primitive of the Euler engine: one launch under one seed.
/// Implement it for a device handle and [`EulerBackend`] follows through
/// `kernel_euler_backend!`; the host handles implement [`EulerBackend`]
/// directly.
pub trait EulerKernel<T: FloatExt>: Backend {
  /// Paths `first .. first + m` of the launch stream seeded by `seed`, as an
  /// `m × n` matrix whose column 0 is the initial value. The kernels hash
  /// `(first + path, step, seed)`, so a batch produced in chunks under one
  /// seed is bit-identical to one launch of the whole batch.
  fn euler_kernel<P: EulerCoefficients<T>>(
    &self,
    process: &P,
    first: usize,
    m: usize,
    seed: u64,
  ) -> Result<Array2<T>, DeviceError>;

  /// Bytes of path data one launch may hold.
  fn batch_budget(&self) -> usize;

  /// The whole batch under `seed`, chunked to the budget. A device may
  /// override it to pipeline the chunks; the result must stay bit-identical.
  fn euler_kernel_batch<P: EulerCoefficients<T>>(
    &self,
    process: &P,
    m: usize,
    seed: u64,
  ) -> Result<Array2<T>, DeviceError> {
    let n = process.grid_points();
    let rows = crate::device::chunk_rows(self.batch_budget(), n, std::mem::size_of::<T>());
    if m <= rows {
      return self.euler_kernel(process, 0, m, seed);
    }
    let mut out = Array2::<T>::zeros((m, n));
    let mut first = 0;
    while first < m {
      let len = rows.min(m - first);
      let chunk = self.euler_kernel(process, first, len, seed)?;
      out
        .slice_mut(ndarray::s![first..first + len, ..])
        .assign(&chunk);
      first += len;
    }
    Ok(out)
  }
}

/// How a backend handle produces Euler paths for the processes it serves:
/// the CPU handles run the process's own sampler, a device handle runs its
/// [`EulerKernel`]. The `try_*` methods report a device failure as a
/// [`DeviceError`]; the plain ones panic with it.
pub trait EulerBackend<T: FloatExt>: Backend {
  /// One path.
  fn try_sample<P: EulerCoefficients<T>>(&self, process: &P) -> Result<Array1<T>, DeviceError>;

  /// `m` paths.
  fn try_euler_paths<P: EulerCoefficients<T>>(
    &self,
    process: &P,
    m: usize,
  ) -> Result<Vec<Array1<T>>, DeviceError>;

  /// `f` over `m` paths, mapped as they are produced, so the batch never has
  /// to fit in memory at once.
  fn try_euler_paths_map<P: EulerCoefficients<T>, R: Send>(
    &self,
    process: &P,
    m: usize,
    f: impl Fn(&Array1<T>) -> R + Sync,
  ) -> Result<Vec<R>, DeviceError>;

  /// The batch as one `m × n` matrix; on a device the launch buffer itself.
  fn try_euler_matrix<P: EulerCoefficients<T>>(
    &self,
    process: &P,
    m: usize,
  ) -> Result<Array2<T>, DeviceError>;

  /// [`try_sample`](Self::try_sample), panicking with the device's error.
  fn euler_sample<P: EulerCoefficients<T>>(&self, process: &P) -> Array1<T> {
    self
      .try_sample(process)
      .unwrap_or_else(crate::device::device_panic)
  }

  /// [`try_euler_paths`](Self::try_euler_paths), panicking with the device's
  /// error; [`Backend::probe`] first turns that failure into a `Result`.
  fn euler_paths<P: EulerCoefficients<T>>(&self, process: &P, m: usize) -> Vec<Array1<T>> {
    self
      .try_euler_paths(process, m)
      .unwrap_or_else(crate::device::device_panic)
  }

  /// [`try_euler_paths_map`](Self::try_euler_paths_map), panicking with the
  /// device's error.
  fn euler_paths_map<P: EulerCoefficients<T>, R: Send>(
    &self,
    process: &P,
    m: usize,
    f: impl Fn(&Array1<T>) -> R + Sync,
  ) -> Vec<R> {
    self
      .try_euler_paths_map(process, m, f)
      .unwrap_or_else(crate::device::device_panic)
  }
}

/// A host handle samples through the process's own sampler, chunked the way
/// `ProcessExt` chunks, so its streams are those of the process.
macro_rules! host_euler_backend {
  ($handle:ty) => {
    impl<T: FloatExt> EulerBackend<T> for $handle {
      fn try_sample<P: EulerCoefficients<T>>(&self, process: &P) -> Result<Array1<T>, DeviceError> {
        Ok(process.host_sample())
      }

      fn try_euler_paths<P: EulerCoefficients<T>>(
        &self,
        process: &P,
        m: usize,
      ) -> Result<Vec<Array1<T>>, DeviceError> {
        Ok(sample_par_chunked(process, m))
      }

      fn try_euler_paths_map<P: EulerCoefficients<T>, R: Send>(
        &self,
        process: &P,
        m: usize,
        f: impl Fn(&Array1<T>) -> R + Sync,
      ) -> Result<Vec<R>, DeviceError> {
        Ok(sample_map_chunked(process, m, f))
      }

      fn try_euler_matrix<P: EulerCoefficients<T>>(
        &self,
        process: &P,
        m: usize,
      ) -> Result<Array2<T>, DeviceError> {
        let rows = sample_par_chunked(process, m);
        let n = rows.first().map_or(process.grid_points(), |r| r.len());
        let mut out = Array2::<T>::zeros((m, n));
        for (i, row) in rows.iter().enumerate() {
          out.row_mut(i).assign(row);
        }
        Ok(out)
      }
    }
  };
}

host_euler_backend!(Cpu);
#[cfg(feature = "accelerate")]
host_euler_backend!(crate::device::Accelerate);

/// A device kernel is an Euler backend: one seed per call, chunks to the
/// handle's budget, the map applied per chunk in parallel. One impl per
/// handle rather than a blanket one, which coherence would not allow beside
/// the host impls above.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cubecl"))]
macro_rules! kernel_euler_backend {
  ($handle:ty, [$($gen:tt)*] $scalar:ty) => {
    impl<$($gen)*> EulerBackend<$scalar> for $handle {
    fn try_sample<P: EulerCoefficients<$scalar>>(&self, process: &P) -> Result<Array1<$scalar>, DeviceError> {
      let seed = process.device_seed();
      Ok(<Self as EulerKernel<$scalar>>::euler_kernel(self, process, 0, 1, seed)?.row(0).to_owned())
    }

    fn try_euler_paths<P: EulerCoefficients<$scalar>>(
      &self,
      process: &P,
      m: usize,
    ) -> Result<Vec<Array1<$scalar>>, DeviceError> {
      let seed = process.device_seed();
      Ok(
        <Self as EulerKernel<$scalar>>::euler_kernel_batch(self, process, m, seed)?
          .outer_iter()
          .map(|row| row.to_owned())
          .collect(),
      )
    }

    fn try_euler_paths_map<P: EulerCoefficients<$scalar>, R: Send>(
      &self,
      process: &P,
      m: usize,
      f: impl Fn(&Array1<$scalar>) -> R + Sync,
    ) -> Result<Vec<R>, DeviceError> {
      use rayon::prelude::*;
      let seed = process.device_seed();
      let rows = crate::device::chunk_rows(
        <Self as EulerKernel<$scalar>>::batch_budget(self),
        process.grid_points(),
        std::mem::size_of::<$scalar>(),
      );
      let mut out = Vec::with_capacity(m);
      let mut first = 0;
      while first < m {
        let len = rows.min(m - first);
        let chunk: Vec<Array1<$scalar>> = <Self as EulerKernel<$scalar>>::euler_kernel(self, process, first, len, seed)?
          .outer_iter()
          .map(|row| row.to_owned())
          .collect();
        out.extend(chunk.par_iter().map(&f).collect::<Vec<R>>());
        first += len;
      }
      Ok(out)
    }

    fn try_euler_matrix<P: EulerCoefficients<$scalar>>(
      &self,
      process: &P,
      m: usize,
    ) -> Result<Array2<$scalar>, DeviceError> {
      <Self as EulerKernel<$scalar>>::euler_kernel_batch(self, process, m, process.device_seed())
    }
    }
  };
}

#[cfg(feature = "metal")]
kernel_euler_backend!(crate::device::Metal, [] f32);
#[cfg(any(feature = "cubecl-cuda", feature = "cubecl-wgpu"))]
kernel_euler_backend!(crate::device::Cubecl<Rt>, [Rt: crate::euler::cubecl::CubeclRuntime] f32);
#[cfg(feature = "cuda")]
kernel_euler_backend!(crate::device::Cuda, [T: FloatExt] T);

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

  fn host_sample(&self) -> Array1<T> {
    let out = self.sampler().sample();
    self.advance_chunk_seed();
    out
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

  fn host_sample(&self) -> Array1<T> {
    let out = self.sampler().sample();
    self.advance_chunk_seed();
    out
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

  fn host_sample(&self) -> Array1<T> {
    let out = self.sampler().sample();
    self.advance_chunk_seed();
    out
  }
}

macro_rules! try_sample_matrix {
  ($ty:ident) => {
    impl<T: FloatExt, S: SeedExt, B: EulerBackend<T>> $ty<T, S, B> {
      /// The batch as one `m × n` matrix: on a device back-end the launch
      /// buffer itself, without a re-layout into rows. The row form is
      /// [`ProcessExt::try_sample_par`], the single path
      /// [`ProcessExt::try_sample`].
      pub fn try_sample_matrix(&self, m: usize) -> Result<Array2<T>, DeviceError> {
        self.backend.try_euler_matrix(self, m)
      }
    }
  };
}

try_sample_matrix!(Gbm);
try_sample_matrix!(Ou);
try_sample_matrix!(Cir);

#[cfg(any(feature = "cubecl-cuda", feature = "cubecl-wgpu"))]
pub mod cubecl;
#[cfg(feature = "cuda")]
pub mod cuda;
// The generated C artifacts have no consumer until `cuda` or `metal` renders
// a kernel from them; the declarations, the family codes and the host step
// stay compiled either way, so a family is checked without a GPU.
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
pub(crate) mod families;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub(crate) mod kernel;
#[cfg(feature = "metal")]
pub mod metal;

#[cfg(test)]
mod tests;

/// A single-precision device refuses an `f64` process at compile time.
///
/// ```compile_fail,E0277
/// use stochastic_rs_core::simd_rng::Unseeded;
/// use stochastic_rs_stochastic::device::Metal;
/// use stochastic_rs_stochastic::diffusion::gbm::Gbm;
/// use stochastic_rs_stochastic::traits::ProcessExt;
///
/// let gbm = Gbm::<f64, _>::new(0.05, 0.2, 16, None, None, Unseeded);
/// let _ = gbm.on::<Metal>().sample();
/// ```
#[cfg(feature = "metal")]
pub mod precision_guard {}

#[cfg(feature = "python")]
pub mod python {
  //! Python surface of the device layer: probing a device and choosing the
  //! ordinal. Sampling on a device goes through the process classes'
  //! `device=` argument.

  use pyo3::prelude::*;

  /// Opens the named device (`"cpu"`, `"accelerate"`, `"cuda"`, `"metal"`,
  /// `"cubecl-cuda"`, `"cubecl-wgpu"`, optionally with `:ordinal`) and describes it
  /// as a dict with `backend`, `name`, `precisions` and `ordinal`; raises
  /// `RuntimeError` with the device's own message when it cannot be used,
  /// `ValueError` for a device this build does not carry.
  #[pyfunction]
  pub fn probe_device<'py>(
    py: Python<'py>,
    device: &str,
  ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let info = crate::python_device::Device::parse_name(device)?.probe()?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("backend", info.backend)?;
    d.set_item("name", info.name)?;
    d.set_item("precisions", info.precisions.to_vec())?;
    d.set_item("ordinal", info.ordinal)?;
    Ok(d)
  }
}
