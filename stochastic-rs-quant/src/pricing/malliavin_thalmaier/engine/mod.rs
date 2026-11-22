//! The engine computing Malliavin–Thalmaier Greeks from simulated multi-asset paths.
//!
//! See Kohatsu-Higa & Yasuda (2008), §6 (Theorem 6.1).

mod cross_gamma_vega;
mod delta;
mod greeks;
mod localization;
mod payoff;

#[cfg(test)]
mod tests;

pub use greeks::MtGreeks;
pub use payoff::MtPayoff;
