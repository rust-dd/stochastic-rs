//! # Training device
//!
//! candle runs the surrogates on the CPU by default; the `metal` and `cuda`
//! cargo features of this crate turn on the matching candle back-ends, and
//! [`best_available`] picks the fastest device the build can reach. Every
//! constructor in [`crate::volatility`] takes a `&Device`, so a training run
//! moves to the GPU by passing this device instead of `Device::Cpu` — the
//! weights are stored device-independently (`safetensors`), so a model trained
//! on one device loads on any other.

use anyhow::Result;
use candle_core::Device;

/// The fastest device this build can use: CUDA when compiled with the `cuda`
/// feature and a device is present, Metal when compiled with the `metal`
/// feature on macOS, the CPU otherwise. Errors only when a GPU back-end is
/// compiled in but its device fails to initialise.
pub fn best_available() -> Result<Device> {
  #[cfg(feature = "cuda")]
  {
    if candle_core::utils::cuda_is_available() {
      return Ok(Device::new_cuda(0)?);
    }
  }
  #[cfg(feature = "metal")]
  {
    if candle_core::utils::metal_is_available() {
      return Ok(Device::new_metal(0)?);
    }
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
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
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
