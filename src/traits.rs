//! Umbrella re-export hub: every documented trait each sub-crate exports, prelude-excluded ones
//! included. The quant half is derivable, and `tests/prelude_completeness.rs` fails on a drop:
//!
//! ```text
//! diff <(grep '^pub use \(calibration\|instrument\|pricing\|short_rate\|time\)::' \
//!          stochastic-rs-quant/src/traits.rs | sed 's/.*:://;s/;//' | sort) \
//!      <(grep '^pub use stochastic_rs_quant::traits::' src/traits.rs \
//!          | sed 's/.*:://;s/;//' | sort)
//! ```

#[cfg(feature = "ai")]
pub use stochastic_rs_ai::traits::SurrogateModel;
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
pub use stochastic_rs_distributions::traits::HypothesisTest;
pub use stochastic_rs_distributions::traits::Program;
pub use stochastic_rs_distributions::traits::ProgramError;
pub use stochastic_rs_distributions::traits::RealExt;
pub use stochastic_rs_distributions::traits::SimdDistribution;
pub use stochastic_rs_distributions::traits::SimdFloatExt;
pub use stochastic_rs_distributions::traits::SimdKernel;
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
pub use stochastic_rs_stats::traits::DiffusionModel;
pub use stochastic_rs_stats::traits::FractalDimEstimator;
pub use stochastic_rs_stats::traits::HurstEstimator;
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
pub use stochastic_rs_stochastic::traits::ProcessExt;
pub use stochastic_rs_stochastic::traits::TwoDimensional;
pub use stochastic_rs_stochastic::traits::VariableDimensional;
pub use stochastic_rs_stochastic::volterra::VolterraKernel;
