use super::Backend;
use super::HostBackend;
#[cfg(device_backend)]
use super::env::DEFAULT_BATCH_BUDGET_BYTES;
#[cfg(device_backend)]
use super::env::env_budget;
#[cfg(device_backend)]
use super::env::env_ordinal;
use super::error::DeviceError;
use super::error::DeviceInfo;
use super::sealed;

/// CPU backend — the default `B` for every process.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cpu;

#[cfg(feature = "cuda")]
device_handle!(
  /// cudarc + cuFFT + NVRTC Philox.
  Cuda,
  "Which device to open: the CUDA device ordinal."
);

#[cfg(metal_backend)]
device_handle!(
  /// Metal compute in `f32` on Apple Silicon with macOS 15 or newer; each thread that samples
  /// keeps its own command queue, kernel library and buffers until it exits.
  Metal,
  "Which device to open: the index into `Device::all()`, `0` being the system default."
);

/// Apple vDSP / AMX (FFI system framework, macOS).
#[cfg(accelerate_backend)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Accelerate;

impl sealed::Sealed for Cpu {}

#[cfg(accelerate_backend)]
impl sealed::Sealed for Accelerate {}

impl Backend for Cpu {
  fn probe(&self) -> Result<DeviceInfo, DeviceError> {
    Ok(DeviceInfo::host("Cpu", "host CPU (SIMD)"))
  }
}

#[cfg(feature = "cuda")]
impl Backend for Cuda {
  fn probe(&self) -> Result<DeviceInfo, DeviceError> {
    crate::euler::cuda::probe(self.ordinal)
  }
}

#[cfg(metal_backend)]
impl Backend for Metal {
  fn probe(&self) -> Result<DeviceInfo, DeviceError> {
    super::metal::probe(self.ordinal)
  }
}

#[cfg(accelerate_backend)]
impl Backend for Accelerate {
  fn probe(&self) -> Result<DeviceInfo, DeviceError> {
    Ok(DeviceInfo::host("Accelerate", "host CPU (Apple vDSP)"))
  }
}

impl HostBackend for Cpu {}

#[cfg(accelerate_backend)]
impl HostBackend for Accelerate {}
