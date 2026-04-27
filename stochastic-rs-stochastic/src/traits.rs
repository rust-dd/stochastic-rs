//! # Stochastic process traits
//!
//! Organised as focused submodules: [`process`] (`ProcessExt` + dimensional
//! markers) and [`malliavin`] (finite-difference Malliavin sensitivities).
//! Upstream traits are re-exported so call-sites can write
//! `crate::traits::FloatExt` without reaching into `stochastic_rs_distributions`.

pub mod malliavin;
pub mod process;

pub use malliavin::Malliavin2DExt;
pub use malliavin::MalliavinExt;
pub use process::CurveOutput;
pub use process::MultiDimensional;
pub use process::OneDimensional;
pub use process::ProcessExt;
pub use process::TwoDimensional;
#[cfg(feature = "python")]
pub use stochastic_rs_distributions::traits::CallableDist;
pub use stochastic_rs_distributions::traits::DistributionExt;
pub use stochastic_rs_distributions::traits::DistributionSampler;
pub use stochastic_rs_distributions::traits::FloatExt;
pub use stochastic_rs_distributions::traits::Fn1D;
pub use stochastic_rs_distributions::traits::Fn2D;
pub use stochastic_rs_distributions::traits::SimdFloatExt;
