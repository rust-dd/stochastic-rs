//! The `device=` argument of the device-capable Python classes: parsed,
//! checked against the build and the precision, and probed at construction,
//! so a class that exists samples on a device that works.

use pyo3::PyResult;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;

use crate::device::Backend;
use crate::device::Cpu;

/// Where a Python-side process samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
  Cpu,
  Accelerate,
  CudaNative,
  MetalNative,
  CubeCl,
}

impl Device {
  /// Parses `device=`: `None` and `"cpu"` are the host, `"gpu"` the first
  /// compiled device back-end (native CUDA, then native Metal, then CubeCL).
  /// A device this build does not carry is a `ValueError` with a rebuild
  /// hint, a single-precision device under a `float64` process is a
  /// `ValueError` asking for `dtype="f32"`, and a device that is compiled in
  /// but cannot be opened is a `RuntimeError` with the runtime's own message.
  pub fn parse(name: Option<&str>, dtype: &str) -> PyResult<Self> {
    let name = name.unwrap_or("cpu").to_ascii_lowercase();
    let device = match name.as_str() {
      "cpu" => Device::Cpu,
      "accelerate" => Device::Accelerate,
      "cuda-native" | "cuda_native" => Device::CudaNative,
      "metal" => Device::MetalNative,
      "cubecl" => Device::CubeCl,
      "gpu" => Device::first_gpu()?,
      other => {
        return Err(PyValueError::new_err(format!(
          "unknown device {other:?}; use cpu, gpu, cuda-native, metal, cubecl or accelerate"
        )));
      }
    };
    device.check_compiled()?;
    if device.single_precision() && dtype != "f32" {
      return Err(PyValueError::new_err(format!(
        "{} computes in single precision; pass dtype=\"f32\"",
        device.name()
      )));
    }
    device.probe()?;
    Ok(device)
  }

  fn first_gpu() -> PyResult<Self> {
    if cfg!(feature = "cuda-native") {
      Ok(Device::CudaNative)
    } else if cfg!(feature = "metal") {
      Ok(Device::MetalNative)
    } else if cfg!(any(feature = "cubecl-cuda", feature = "cubecl-wgpu")) {
      Ok(Device::CubeCl)
    } else {
      Err(PyValueError::new_err(
        "this build has no GPU runtime; rebuild with the cuda-native, metal, cubecl-cuda or cubecl-wgpu feature",
      ))
    }
  }

  fn check_compiled(self) -> PyResult<()> {
    let (compiled, what, feature) = match self {
      Device::Cpu => (true, "", ""),
      Device::Accelerate => (
        cfg!(feature = "accelerate"),
        "Accelerate back-end",
        "accelerate",
      ),
      Device::CudaNative => (
        cfg!(feature = "cuda-native"),
        "native CUDA runtime",
        "cuda-native",
      ),
      Device::MetalNative => (cfg!(feature = "metal"), "native Metal runtime", "metal"),
      Device::CubeCl => (
        cfg!(any(feature = "cubecl-cuda", feature = "cubecl-wgpu")),
        "CubeCL runtime",
        "cubecl-cuda or cubecl-wgpu",
      ),
    };
    if compiled {
      Ok(())
    } else {
      Err(PyValueError::new_err(format!(
        "this build has no {what}; rebuild with the {feature} feature"
      )))
    }
  }

  /// Metal and CubeCL kernels compute in `f32` only.
  pub fn single_precision(self) -> bool {
    matches!(self, Device::MetalNative | Device::CubeCl)
  }

  /// The name `device=` accepts for this variant.
  pub fn name(self) -> &'static str {
    match self {
      Device::Cpu => "cpu",
      Device::Accelerate => "accelerate",
      Device::CudaNative => "cuda-native",
      Device::MetalNative => "metal",
      Device::CubeCl => "cubecl",
    }
  }

  fn probe(self) -> PyResult<()> {
    let info = match self {
      Device::Cpu => Cpu::probe(),
      #[cfg(feature = "accelerate")]
      Device::Accelerate => crate::device::Accelerate::probe(),
      #[cfg(feature = "cuda-native")]
      Device::CudaNative => crate::device::CudaNative::probe(),
      #[cfg(feature = "metal")]
      Device::MetalNative => crate::device::MetalNative::probe(),
      #[cfg(any(feature = "cubecl-cuda", feature = "cubecl-wgpu"))]
      Device::CubeCl => crate::device::CubeCl::probe(),
      #[allow(unreachable_patterns)]
      _ => unreachable!("check_compiled rejects the devices this build lacks"),
    };
    info
      .map(|_| ())
      .map_err(|e| PyRuntimeError::new_err(e.to_string()))
  }
}
