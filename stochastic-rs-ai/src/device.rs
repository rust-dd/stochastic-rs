//! # Training device
//!
//! candle runs the surrogates on the CPU by default. The `metal` cargo
//! feature of this crate turns on candle's Metal back-end; CUDA comes from
//! enabling `candle-core/cuda` in the consuming manifest (candle-kernels needs
//! a CUDA toolkit to build, so this crate cannot carry it as a feature). In
//! both cases [`best_available`] detects the device at run time. Every
//! constructor in [`crate::volatility`] takes a `&Device`, so a training run
//! moves to the GPU by passing this device instead of `Device::Cpu` — the
//! weights are stored device-independently (`safetensors`), so a model trained
//! on one device loads on any other.

use anyhow::Result;
use candle_core::Device;

/// The fastest device this build can reach: CUDA when candle's CUDA back-end
/// is compiled in and a device is present, Metal when candle's Metal
/// back-end is compiled in (this crate's `metal` feature) on macOS, the CPU
/// otherwise. Errors only when a compiled-in back-end fails to initialise
/// its device.
pub fn best_available() -> Result<Device> {
  if candle_core::utils::cuda_is_available() {
    return Ok(Device::new_cuda(0)?);
  }
  if candle_core::utils::metal_is_available() {
    return Ok(Device::new_metal(0)?);
  }
  Ok(Device::Cpu)
}

/// Human-readable name of a device, for logs and reports.
pub fn describe(device: &Device) -> &'static str {
  match device {
    Device::Cpu => "cpu",
    Device::Cuda(_) => "cuda",
    Device::Metal(_) => "metal",
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::volatility::common::TrainConfig;
  use crate::volatility::common::synthetic_surface_dataset;
  use crate::volatility::heston;
  use crate::volatility::heston::HestonNn;

  #[test]
  fn best_available_is_a_usable_device() {
    let device = best_available().unwrap();
    let name = describe(&device);
    assert!(["cpu", "cuda", "metal"].contains(&name));
    #[cfg(not(feature = "metal"))]
    assert_eq!(name, "cpu");
  }

  /// A training step on the selected device produces the same kind of
  /// report as the CPU path (and, with a GPU feature compiled in, runs
  /// there).
  #[test]
  fn training_runs_on_the_best_device() {
    let device = best_available().unwrap();
    let (params, surfaces) = synthetic_surface_dataset(
      &heston::PARAM_LB,
      &heston::PARAM_UB,
      64,
      heston::OUTPUT_DIM,
      4,
    );
    let mut model = HestonNn::new(&device).unwrap();
    let cfg = TrainConfig {
      epochs: 2,
      ..TrainConfig::default()
    };
    let report = model.train(&params, &surfaces, &cfg).unwrap();
    assert_eq!(report.epochs.len(), 2);
    assert!(report.epochs[1].val_rmse.is_finite());
  }
}
