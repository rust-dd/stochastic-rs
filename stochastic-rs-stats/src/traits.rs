//! Re-exports of upstream traits so `crate::traits::Foo` continues to resolve
//! inside the stats sub-crate's source files.

pub use stochastic_rs_distributions::traits::DistributionExt;
pub use stochastic_rs_distributions::traits::DistributionSampler;
pub use stochastic_rs_distributions::traits::FloatExt;
pub use stochastic_rs_distributions::traits::Fn1D;
pub use stochastic_rs_distributions::traits::Fn2D;
pub use stochastic_rs_distributions::traits::SimdFloatExt;
pub use stochastic_rs_stochastic::traits::Malliavin2DExt;
pub use stochastic_rs_stochastic::traits::MalliavinExt;
pub use stochastic_rs_stochastic::traits::ProcessExt;
