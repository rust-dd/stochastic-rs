//! # Stochastic process traits
//!
//! [`process`] holds `ProcessExt` and the dimensional markers, [`sampler`] holds [`PathSampler`];
//! upstream traits are re-exported so a call site writes `crate::traits::FloatExt`.

pub mod process;
pub mod sampler;

mod sealed {
  #[diagnostic::on_unimplemented(
    message = "`{Self}` cannot implement the sealed traits `ProcessExt` and `PathSampler`",
    note = "wrap or compose an in-tree process; a new process is added inside stochastic-rs-stochastic"
  )]
  pub trait Sealed {}
}

pub(crate) use sealed::Sealed;

pub use process::ComplexPathOutput;
pub use process::CurveOutput;
pub use process::MultiDimensional;
pub use process::OneDimensional;
pub use process::ProcessExt;
pub use process::TwoDimensional;
pub use process::VariableDimensional;
pub use sampler::PathSampler;
#[cfg(feature = "python")]
pub use stochastic_rs_distributions::traits::CallableDist;
pub use stochastic_rs_distributions::traits::DistributionExt;
pub use stochastic_rs_distributions::traits::DistributionSampler;
pub use stochastic_rs_distributions::traits::Expr;
pub use stochastic_rs_distributions::traits::FloatExt;
pub use stochastic_rs_distributions::traits::Fn1D;
pub use stochastic_rs_distributions::traits::Fn2D;
pub use stochastic_rs_distributions::traits::Grid2D;
pub use stochastic_rs_distributions::traits::Program;
pub use stochastic_rs_distributions::traits::ProgramError;
pub use stochastic_rs_distributions::traits::RealExt;
pub use stochastic_rs_distributions::traits::SimdFloatExt;
