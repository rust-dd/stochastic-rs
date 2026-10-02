//! # Traits — umbrella re-export hub.
//!
//! Mirrors every trait each sub-crate exports from its own `traits` module; a trait kept out of
//! [`crate::prelude`] (`ShortRatePricer`, `VanillaEuropeanCall`, `GreeksExt`) still resolves here.
//!
//! The quant half of the mirror is derivable, so a future omission is
//! measurable rather than a matter of reading:
//!
//! ```text
//! diff <(grep '^pub use \(calibration\|instrument\|pricing\|short_rate\|time\)::' \
//!          stochastic-rs-quant/src/traits.rs | sed 's/.*:://;s/;//' | sort) \
//!      <(grep '^pub use stochastic_rs_quant::traits::' src/traits.rs \
//!          | sed 's/.*:://;s/;//' | sort)
//! ```
//!
//! `tests/prelude_completeness.rs` names the prelude-excluded traits
//! explicitly, so dropping one from this hub is a compile error there.

pub use stochastic_rs_copulas::traits::BivariateExt;
pub use stochastic_rs_copulas::traits::MultivariateExt;
pub use stochastic_rs_copulas::traits::TailDependence;
pub use stochastic_rs_distributions::traits::DistributionExt;
pub use stochastic_rs_distributions::traits::DistributionSampler;
pub use stochastic_rs_distributions::traits::Expr;
pub use stochastic_rs_distributions::traits::FloatExt;
pub use stochastic_rs_distributions::traits::Fn1D;
pub use stochastic_rs_distributions::traits::Fn2D;
pub use stochastic_rs_distributions::traits::Grid2D;
pub use stochastic_rs_distributions::traits::Program;
pub use stochastic_rs_distributions::traits::RealExt;
pub use stochastic_rs_distributions::traits::SimdFloatExt;
pub use stochastic_rs_quant::traits::CalibrationResult;
pub use stochastic_rs_quant::traits::Calibrator;
pub use stochastic_rs_quant::traits::Greeks;
pub use stochastic_rs_quant::traits::GreeksExt;
pub use stochastic_rs_quant::traits::Instrument;
pub use stochastic_rs_quant::traits::InstrumentExt;
pub use stochastic_rs_quant::traits::ModelPricer;
pub use stochastic_rs_quant::traits::PricingEngine;
pub use stochastic_rs_quant::traits::PricingResult;
pub use stochastic_rs_quant::traits::ShortRatePricer;
pub use stochastic_rs_quant::traits::StandardResult;
pub use stochastic_rs_quant::traits::TimeExt;
pub use stochastic_rs_quant::traits::ToModel;
pub use stochastic_rs_quant::traits::ToShortRateModel;
pub use stochastic_rs_quant::traits::VanillaEuropeanCall;
pub use stochastic_rs_stats::fractal_dim::FractalDimEstimator;
pub use stochastic_rs_stats::hurst::HurstEstimator;
pub use stochastic_rs_stats::mle::DiffusionModel;
pub use stochastic_rs_stats::traits::HypothesisTest;
pub use stochastic_rs_stochastic::device::Backend;
pub use stochastic_rs_stochastic::device::Cpu;
pub use stochastic_rs_stochastic::device::FgnBackend;
pub use stochastic_rs_stochastic::device::HostBackend;
pub use stochastic_rs_stochastic::device::SheetBackend;
pub use stochastic_rs_stochastic::euler::EulerBackend;
pub use stochastic_rs_stochastic::traits::ComplexPathOutput;
pub use stochastic_rs_stochastic::traits::CurveOutput;
pub use stochastic_rs_stochastic::traits::MultiDimensional;
pub use stochastic_rs_stochastic::traits::OneDimensional;
pub use stochastic_rs_stochastic::traits::PathSampler;
pub use stochastic_rs_stochastic::traits::ProcessExt;
pub use stochastic_rs_stochastic::traits::TwoDimensional;
pub use stochastic_rs_stochastic::traits::VariableDimensional;
pub use stochastic_rs_stochastic::volterra::VolterraKernel;
