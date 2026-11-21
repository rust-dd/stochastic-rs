//! Quant traits by area: [`pricing`], [`instrument`], [`time`], [`short_rate`] and [`calibration`],
//! with the foundation crates' `RealExt`, `FloatExt` and `ProcessExt` reachable here.

pub mod calibration;
pub mod instrument;
pub mod pricing;
pub mod short_rate;
pub mod time;

pub use calibration::CalibrationResult;
pub use calibration::Calibrator;
pub use calibration::ToModel;
pub use calibration::ToShortRateModel;
pub use instrument::Instrument;
pub use instrument::InstrumentExt;
pub use instrument::PricingEngine;
pub use instrument::PricingResult;
pub use instrument::StandardResult;
pub use pricing::Greeks;
pub use pricing::GreeksExt;
pub use pricing::ModelPricer;
pub use pricing::VanillaEuropeanCall;
pub use short_rate::ShortRatePricer;
pub use stochastic_rs_distributions::traits::FloatExt;
pub use stochastic_rs_distributions::traits::RealExt;
pub use stochastic_rs_stochastic::traits::ProcessExt;
pub use time::TimeExt;
