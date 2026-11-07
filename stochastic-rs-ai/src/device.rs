//! Training runs on the CPU, on Metal with this crate's `metal` feature, or on CUDA when the
//! consumer enables `candle-core/cuda` (it needs a CUDA toolkit); saved weights load anywhere.

use anyhow::Result;
use candle_core::Device;

/// The fastest compiled-in device that is present (CUDA, then Metal, then the CPU); errors only
/// when a compiled-in back-end fails to initialise.
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

  /// A training step runs on the selected device and reports as the CPU path does.
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
