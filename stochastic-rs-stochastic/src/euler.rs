//! Device-side Euler–Maruyama path generation: one device thread per path, Philox normals.
//!
//! $$
//! X_{i+1} = X_i + b(X_i)\,\Delta t + \sigma(X_i)\sqrt{\Delta t}\,Z_i,\qquad
//! Z_i \sim \mathcal N(0, 1)
//! $$
//!
//! Only [`EulerBackend`] is stable: <https://stochastic.rust-dd.com/docs/processes#euler-engine>.
//!
//! Reference: Kloeden & Platen (1992), Springer, §10.2, DOI 10.1007/978-3-662-12616-5
//! Reference: Lord et al. (2010), Quant. Finance 10(2), 177–194, DOI 10.1080/14697680802392496

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
