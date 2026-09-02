//! # stochastic-rs-ai
//!
//! Neural surrogate models for stochastic volatility.

pub mod volatility;

/// Surrogate-based calibration into quant's `Calibrator` pipeline.
#[cfg(feature = "quant")]
pub mod calibration;

pub mod device;

/// PyO3 classes and functions (feature `python`, which implies `quant`).
#[cfg(feature = "python")]
pub mod python;

pub use candle_core::Device;
