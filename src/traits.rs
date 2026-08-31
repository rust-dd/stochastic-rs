//! # Traits — umbrella re-export hub.
//!
//! Mirrors every trait each sub-crate exports from its own `traits` module,
//! under the same feature gates. Hub membership is **independent of prelude
//! membership**: a trait kept out of [`crate::prelude`] — because it has no
//! in-tree implementors, or because it is feature-gated and the prelude is
//! deliberately feature-flag-free — still belongs here, and
//! `website/content/docs/concepts/prelude.mdx` tells readers to "reach via
//! `traits::*`" on exactly that basis. `MultivariateExt`, `CallableDist`,
//! `ShortRatePricer`, `VanillaEuropeanCall` and `GreeksExt` are all in that
//! position.
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
#[cfg(feature = "python")]
pub use stochastic_rs_distributions::traits::CallableDist;
pub use stochastic_rs_distributions::traits::DistributionExt;
pub use stochastic_rs_distributions::traits::DistributionSampler;
pub use stochastic_rs_distributions::traits::FloatExt;
pub use stochastic_rs_distributions::traits::Fn1D;
pub use stochastic_rs_distributions::traits::Fn2D;
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
pub use stochastic_rs_stochastic::traits::ComplexPathOutput;
pub use stochastic_rs_stochastic::traits::CurveOutput;
pub use stochastic_rs_stochastic::traits::MultiDimensional;
pub use stochastic_rs_stochastic::traits::OneDimensional;
pub use stochastic_rs_stochastic::traits::PathSampler;
pub use stochastic_rs_stochastic::traits::ProcessExt;
pub use stochastic_rs_stochastic::traits::TwoDimensional;
pub use stochastic_rs_stochastic::traits::VariableDimensional;
pub use stochastic_rs_stochastic::volterra::VolterraKernel;
