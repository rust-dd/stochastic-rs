//! Compile-time sampling backends, picked by `process.on::<Cuda>()` without a runtime branch; a
//! marker exists only where it builds, `Metal` on Apple Silicon macOS and `Accelerate` on macOS.

mod chunks;
mod env;
mod error;
mod fgn_backend;
mod handles;
#[cfg(metal_backend)]
pub(crate) mod metal;
mod sheet_backend;
#[cfg(test)]
mod tests;

pub(crate) use chunks::chunk_rows;
pub(crate) use chunks::device_panic;
pub(crate) use chunks::launch_len;
#[cfg(device_backend)]
pub(crate) use chunks::lru_slot;
pub(crate) use chunks::over_chunks;
#[cfg(device_backend)]
pub(crate) use chunks::rows_within_index_limit;
pub use env::DEFAULT_BATCH_BUDGET_BYTES;
#[cfg(feature = "python")]
pub(crate) use env::env_budget;
#[cfg(feature = "python")]
pub(crate) use env::env_ordinal;
pub use error::DeviceError;
pub use error::DeviceInfo;
pub use fgn_backend::FgnBackend;
#[cfg(accelerate_backend)]
pub use handles::Accelerate;
pub use handles::Cpu;
#[cfg(feature = "cuda")]
pub use handles::Cuda;
#[cfg(metal_backend)]
pub use handles::Metal;
pub use sheet_backend::SheetBackend;

/// How many per-size states (plans, buffers) each fGN and sheet device cache keeps, and how many
/// outputs the Metal fGN pool holds; a build without a device has no caller.
#[cfg_attr(not(device_backend), allow(dead_code))]
pub(crate) const CACHE_SLOTS: usize = 4;

/// Frees the fGN FFT workspace and, with `accelerate`, the vDSP setups of this thread and of every
/// thread of the current rayon pool, waiting for each; a scratch in use stays, none shrinks alone.
pub fn release_host_scratch() {
  let release = || {
    stochastic_rs_distributions::release_fgn_scratch();
    #[cfg(accelerate_backend)]
    crate::noise::fgn::release_accelerate_scratch();
  };
  release();
  rayon::broadcast(|_| release());
}

/// A compile-time device marker, sealed to the handles this module declares: `probe` is the one run-time
/// question it answers, the sampling stays a compile-time choice (`.on::<B>()`, zero runtime branching).
pub trait Backend: sealed::Sealed + Clone + Send + Sync {
  /// Opens the device behind this marker and describes it, or says why it
  /// cannot be used (no device, runtime missing, kernels failing to
  /// compile). The CPU devices are always `Ok`. A `sample*` call on a device
  /// that fails this probe panics with the same error; the `try_*` calls
  /// return it.
  fn probe(&self) -> Result<DeviceInfo, DeviceError>;
}

mod sealed {
  #[diagnostic::on_unimplemented(
    message = "`{Self}` cannot implement the sealed trait `Backend`",
    note = "a new device is added inside stochastic-rs-stochastic; today's handles are `Cpu`, `Accelerate`, `Cuda` and `Metal`"
  )]
  pub trait Sealed {}
}

/// Host capability: the process samples on the CPU through its own sampler. Every process accepts
/// at least the host devices in `on::<B2>()`; the family is sealed, so a wider bound later breaks nobody.
pub trait HostBackend: Backend {}
