#![doc = include_str!("../README.md")]

pub mod volatility;

/// Surrogate-based calibration into quant's `Calibrator` pipeline.
#[cfg(feature = "quant")]
pub mod calibration;

pub mod device;

/// PyO3 classes and functions (feature `python`, which implies `quant`).
#[cfg(feature = "python")]
#[doc(hidden)]
pub mod python;

pub use candle_core::Device;
