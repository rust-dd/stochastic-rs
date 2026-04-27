//! # Quant traits — pricing, time, calibration, model bridging.
//!
//! Organised as focused submodules: [`pricing`] (`PricerExt` / `ModelPricer` /
//! `GreeksExt`), [`time`] (`TimeExt`), and [`calibration`] (`Calibrator` /
//! `CalibrationResult` / `ToModel` / `ToShortRateModel`). Upstream traits are
//! re-exported so call-sites can write `crate::traits::FloatExt` without
//! reaching into the foundation crates directly.

pub mod calibration;
pub mod pricing;
pub mod time;

pub use stochastic_rs_distributions::traits::DistributionExt;
pub use stochastic_rs_distributions::traits::DistributionSampler;
pub use stochastic_rs_distributions::traits::FloatExt;
pub use stochastic_rs_distributions::traits::Fn1D;
pub use stochastic_rs_distributions::traits::Fn2D;
pub use stochastic_rs_distributions::traits::SimdFloatExt;
pub use stochastic_rs_stochastic::traits::CurveOutput;
pub use stochastic_rs_stochastic::traits::Malliavin2DExt;
pub use stochastic_rs_stochastic::traits::MalliavinExt;
pub use stochastic_rs_stochastic::traits::MultiDimensional;
pub use stochastic_rs_stochastic::traits::OneDimensional;
pub use stochastic_rs_stochastic::traits::ProcessExt;
pub use stochastic_rs_stochastic::traits::TwoDimensional;

pub use stochastic_rs_copulas::traits::BivariateExt;
#[cfg(feature = "openblas")]
pub use stochastic_rs_copulas::traits::MultivariateExt;

pub use calibration::CalibrationResult;
pub use calibration::Calibrator;
pub use calibration::ToModel;
pub use calibration::ToShortRateModel;
pub use pricing::GreeksExt;
pub use pricing::ModelPricer;
pub use pricing::PricerExt;
pub use time::TimeExt;
