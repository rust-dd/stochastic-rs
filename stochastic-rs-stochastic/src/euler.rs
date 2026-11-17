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
//!   Euler–Maruyama recursion in the kernel and Box–Muller normals from
//!   Philox-4x32-10 on `(path, step, site, index)`: `Cuda` (feature `cuda`: cudarc +
//!   NVRTC, `f32` or `f64` after `T`) and `Metal` (feature `metal`:
//!   Rust device kernels compiled by metal-oxide, `f32`).
//!   `sample_par` is one launch for all `m` paths; `sample` launches one path.
//!
//! The device seed is drawn from the process's own seed source, so the same
//! `Deterministic` seed value gives the same device paths, consecutive calls
//! advance the stream and an `Unseeded` process draws fresh entropy, exactly
//! as on the host. The device
//! kernels share one generator keyed on the whole seed, so the device back-ends agree with each
//! other seed for seed up to libm rounding; the host path is the process's
//! own stream, so CPU and device paths agree in distribution, not bit for bit.
//!
//! A process joins the engine by describing its coefficients as an
//! [`EulerSpec`] through [`EulerCoefficients`].
//!
//! ## What is public here
//!
//! [`EulerBackend`] is the capability a generic bound names — `B:
//! EulerBackend<T>` says a process runs on the device `B` through this
//! engine — and it is the one item of this module a caller writes. Everything
//! else — the family specification [`EulerSpec`], the launch hooks
//! [`EulerCoefficients`] and [`EulerSystem`], the lift, table and program
//! specifications, the slot caps, the kernel-implementor trait
//! [`EulerKernel`] — is the engine's own machinery: public so the crate's
//! sub-modules, tests and probes can reach it, hidden from the rendered
//! documentation, and not covered by the crate's stability promise. A
//! process reaches a device by declaring a family in this crate's table, not
//! by implementing these traits from outside.
//!
//! References: Kloeden, P. E. & Platen, E. (1992), *Numerical Solution of
//! Stochastic Differential Equations*, Springer, §10.2 (Euler–Maruyama);
//! Lord, R., Koekkoek, R. & van Dijk, D. (2010), *A comparison of biased
//! simulation schemes for stochastic volatility models*, Quantitative Finance
//! 10(2), 177–194 (full truncation, used by the device kernels for CIR).

use ndarray::Array1;
use ndarray::Array2;
use ndarray::Array3;
use stochastic_rs_distributions::traits::Program;

use crate::device::Backend;
use crate::device::Cpu;
use crate::device::DeviceError;
use crate::diffusion::cir::Cir;
use crate::diffusion::gbm::Gbm;
use crate::diffusion::ou::Ou;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Paths one launch may hold: the budget at `per_path` scalars each, cut by
/// [`EulerKernel::index_rows`]. Every engine batch loop sizes chunks here, so none wraps an index.
fn launch_rows<T: FloatExt, K: EulerKernel<T>>(
  kernel: &K,
  per_path: usize,
  spec: EulerSpec<T>,
  steps: usize,
  fgn: Option<FgnSpec<'_, T>>,
) -> usize {
  let rows = crate::device::chunk_rows(kernel.batch_budget(), per_path, std::mem::size_of::<T>());
  kernel.index_rows(rows, spec, steps, fgn)
}

/// The planes a system launch writes and the paths one launch may hold: every plane the family
/// steps counts, which may exceed the `D` the process reads back (Bergomi steps four for two).
fn system_rows<T: FloatExt, K: EulerKernel<T>, const D: usize, P: EulerSystem<T, D>>(
  kernel: &K,
  process: &P,
) -> (usize, usize) {
  let n = process.grid_points();
  let spec = process.euler_spec();
  let (code, _) = spec.encode();
  let planes = families::Family::from_code(code)
    .expect("a declared family")
    .components();
  (
    planes,
    launch_rows(kernel, n * planes, spec, n, process.fgn_spec()),
  )
}

#[cfg(feature = "cuda")]
#[doc(hidden)]
pub mod cuda;
pub(crate) mod families;
#[cfg(any(feature = "cuda", test))]
pub(crate) mod kernel;
#[cfg(metal_backend)]
#[doc(hidden)]
pub mod metal;
#[doc(hidden)]
pub mod philox;

#[cfg(test)]
mod family_parity;
#[cfg(test)]
mod tests;

mod backend;
mod coefficients;
mod jumps;
mod kernel_trait;
mod slots;
mod spec;

pub use backend::EulerBackend;
#[doc(hidden)]
pub use coefficients::EulerCoefficients;
#[doc(hidden)]
pub use coefficients::EulerSystem;
#[doc(hidden)]
pub use coefficients::check_arity;
#[doc(hidden)]
pub use coefficients::draw_seed;
#[doc(hidden)]
pub use coefficients::system_row;
#[doc(hidden)]
pub use jumps::GammaDraws;
#[doc(hidden)]
pub use jumps::JumpSizes;
pub(crate) use jumps::count_fallback;
#[doc(hidden)]
pub use kernel_trait::EulerKernel;
#[cfg_attr(not(device_backend), allow(unused_imports))]
pub(crate) use kernel_trait::check_path_index;
#[cfg_attr(not(device_backend), allow(unused_imports))]
pub(crate) use kernel_trait::launch_scalars;
pub(crate) use slots::CORRELATED_STREAMS;
#[doc(hidden)]
pub use slots::CURVE_SLOTS;
#[doc(hidden)]
pub use slots::FgnSpec;
#[doc(hidden)]
pub use slots::HISTORY_SLOTS;
#[doc(hidden)]
pub use slots::LIFT_SLOTS;
#[doc(hidden)]
pub use slots::LiftSpec;
pub(crate) use slots::PARAM_SLOTS;
#[doc(hidden)]
pub use slots::PROGRAM_SLOTS;
#[doc(hidden)]
pub use slots::ProgramSpec;
#[doc(hidden)]
pub use slots::SERIES_SLOTS;
#[doc(hidden)]
pub use slots::TABLE_SLOTS;
#[doc(hidden)]
pub use slots::TableSpec;
#[cfg_attr(not(device_backend), allow(unused_imports))]
pub(crate) use slots::encode_lift;
#[cfg_attr(not(device_backend), allow(unused_imports))]
pub(crate) use slots::encode_programs;
#[cfg_attr(not(device_backend), allow(unused_imports))]
pub(crate) use slots::flatten_curves;
#[cfg_attr(not(device_backend), allow(unused_imports))]
pub(crate) use slots::history_slot;
pub(crate) use slots::pack_cholesky;
#[cfg_attr(not(device_backend), allow(unused_imports))]
pub(crate) use slots::series_live;
#[cfg_attr(not(device_backend), allow(unused_imports))]
pub(crate) use slots::series_terms;
#[cfg_attr(not(device_backend), allow(unused_imports))]
pub(crate) use slots::table_terms;
#[doc(hidden)]
pub use spec::EulerSpec;

#[cfg(feature = "python")]
#[doc(hidden)]
pub mod python;
