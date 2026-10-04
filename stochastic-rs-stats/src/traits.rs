//! Re-exports of upstream traits so `crate::traits::Foo` resolves inside the stats sub-crate;
//! `HypothesisTest` is defined in distributions, where copulas can implement it too.

pub use stochastic_rs_distributions::traits::DistributionExt;
pub use stochastic_rs_distributions::traits::DistributionSampler;
pub use stochastic_rs_distributions::traits::FloatExt;
pub use stochastic_rs_distributions::traits::Fn1D;
pub use stochastic_rs_distributions::traits::Fn2D;
pub use stochastic_rs_distributions::traits::HypothesisTest;
pub use stochastic_rs_distributions::traits::RealExt;
pub use stochastic_rs_distributions::traits::SimdFloatExt;
pub use stochastic_rs_stochastic::traits::ProcessExt;
