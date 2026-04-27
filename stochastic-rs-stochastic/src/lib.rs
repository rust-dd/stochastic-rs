//! # stochastic-rs-stochastic
//!
//! Stochastic process simulation: 140+ process types implementing `ProcessExt`.

#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

#[macro_use]
mod macros;

pub mod traits;

pub use stochastic_rs_core::simd_rng;
pub use stochastic_rs_distributions as distributions;
pub use stochastic_rs_distributions::traits::DistributionExt;
pub use stochastic_rs_distributions::traits::SimdFloatExt;

pub use crate::traits::Malliavin2DExt;
pub use crate::traits::MalliavinExt;
pub use crate::traits::ProcessExt;

pub mod aliases;
pub mod autoregressive;
pub mod correlation;
pub mod diffusion;
pub mod interest;
pub mod isonormal;
pub mod ito;
pub mod jump;
pub mod malliavin;
pub mod mc;
pub mod noise;
pub mod numerics;
pub mod process;
pub mod rough;
pub mod sde;
pub mod sheet;
pub mod volatility;

/// Default number of time steps
pub const N: usize = 1000;
/// Default initial value
pub const X0: f64 = 0.5;
/// Default spot price for financial models
pub const S0: f64 = 100.0;
/// Default strike price
pub const K: f64 = 100.0;
