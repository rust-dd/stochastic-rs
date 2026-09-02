//! # stochastic-rs-ai
//!
//! Neural surrogate models for stochastic volatility.

pub mod volatility;

/// Surrogate-based calibration into quant's `Calibrator` pipeline.
#[cfg(feature = "quant")]
pub mod calibration;

pub mod device;

pub use candle_core::Device;
