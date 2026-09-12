//! The `device=` argument of the device-capable Python classes: parsed,
//! checked against the build and the precision, and probed at construction,
//! so a class that exists samples on a device that works. A name may carry
//! an ordinal, `"cuda:1"`, `"metal:0"`; without one the handle's default
//! (`STOCHASTIC_RS_DEVICE`, else `0`) applies.

use std::sync::Arc;

use pyo3::PyResult;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_core::simd_rng::SimdRngExt;

use crate::device::Backend;
use crate::device::Cpu;
use crate::device::DeviceInfo;

/// Seed state shared by a Python process and its temporary backend instances.
/// Derived child streams remain independent of the parent and one another.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct SharedSeed(Arc<Deterministic>);

impl SharedSeed {
  pub fn new(seed: u64) -> Self {
    Self(Arc::new(Deterministic::new(seed)))
  }
}

impl SeedExt for SharedSeed {
  fn rng(&self) -> SimdRng {
    self.0.rng()
  }

  fn derive(&self) -> Self {
    Self(Arc::new(self.0.derive()))
  }

  fn rng_ext<R: SimdRngExt>(&self) -> R {
    self.0.rng_ext()
  }

  fn reseed(&self, seed: u64) {
    self.0.reseed(seed);
  }

  fn seed_value(&self) -> u64 {
    self.0.seed_value()
  }
}

/// Where a Python-side process samples, with the device ordinal where the
/// back-end enumerates devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
  Cpu,
  Accelerate,
  Cuda(usize),
  Metal(usize),
}

impl Device {
  /// Parses `device=` for a process of `dtype`: the name is checked against
  /// the build (`ValueError` with a rebuild hint), a single-precision device
  /// under a `float64` process asks for `dtype="f32"` (`ValueError`), and the
  /// device is probed (`RuntimeError` with the runtime's own message).
  pub fn parse(name: Option<&str>, dtype: &str) -> PyResult<Self> {
    let device = Self::parse_name(name.unwrap_or("cpu"))?;
    if device.single_precision() && dtype != "f32" {
      return Err(PyValueError::new_err(format!(
        "{} computes in single precision; pass dtype=\"f32\"",
        device.name()
      )));
    }
    device.probe()?;
    Ok(device)
  }

  /// Parses `name[:ordinal]` and checks that the build carries the back-end;
  /// no probe.
  pub fn parse_name(name: &str) -> PyResult<Self> {
    let lower = name.to_ascii_lowercase();
    let (kind, ordinal) = match lower.split_once(':') {
      Some((k, o)) => {
        let o: usize = o.trim().parse().map_err(|_| {
          PyValueError::new_err(format!(
            "device ordinal must be a non-negative integer, got {o:?}"
          ))
        })?;
        (k.trim().to_string(), Some(o))
      }
      None => (lower.clone(), None),
    };
    let ordinal = |default: usize| ordinal.unwrap_or(default);
    let device = match kind.as_str() {
      "cpu" => Device::Cpu,
      "accelerate" => Device::Accelerate,
      "cuda" => Device::Cuda(ordinal(crate::device::env_ordinal())),
      "metal" => Device::Metal(ordinal(crate::device::env_ordinal())),
      other => {
        return Err(PyValueError::new_err(format!(
          "unknown device {other:?}; use cpu, accelerate, cuda or metal, optionally with :ordinal"
        )));
      }
    };
    device.check_compiled()?;
    Ok(device)
  }

  fn check_compiled(self) -> PyResult<()> {
    let (compiled, what, feature) = match self {
      Device::Cpu => (true, "", ""),
      Device::Accelerate => (
        cfg!(feature = "accelerate"),
        "Accelerate back-end",
        "accelerate",
      ),
      Device::Cuda(_) => (cfg!(feature = "cuda"), "native CUDA runtime", "cuda"),
      Device::Metal(_) => (cfg!(feature = "metal"), "native Metal runtime", "metal"),
    };
    if compiled {
      Ok(())
    } else {
      Err(PyValueError::new_err(format!(
        "this build has no {what}; rebuild with the {feature} feature"
      )))
    }
  }

  /// The Metal kernels compute in `f32` only.
  pub fn single_precision(self) -> bool {
    matches!(self, Device::Metal(_))
  }

  /// The name `device=` accepts for this variant.
  pub fn name(self) -> &'static str {
    match self {
      Device::Cpu => "cpu",
      Device::Accelerate => "accelerate",
      Device::Cuda(_) => "cuda",
      Device::Metal(_) => "metal",
    }
  }

  /// Opens the device and describes it, `RuntimeError` when it cannot be used.
  pub fn probe(self) -> PyResult<DeviceInfo> {
    let info = match self {
      Device::Cpu => Cpu.probe(),
      #[cfg(feature = "accelerate")]
      Device::Accelerate => crate::device::Accelerate.probe(),
      #[cfg(feature = "cuda")]
      Device::Cuda(o) => crate::device::Cuda::new(o).probe(),
      #[cfg(feature = "metal")]
      Device::Metal(o) => crate::device::Metal::new(o).probe(),
      #[allow(unreachable_patterns)]
      _ => unreachable!("check_compiled rejects the devices this build lacks"),
    };
    info.map_err(|e| PyRuntimeError::new_err(e.to_string()))
  }
}
