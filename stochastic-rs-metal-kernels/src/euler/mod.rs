//! Rust Euler entrypoints instantiated from the shared family declarations.

mod args;
mod draws;
mod families;
mod frame;
mod jumps;
mod kat;
mod memory;
#[allow(dead_code, reason = "the host's maps and key are not drawn here")]
#[path = "../../../stochastic-rs-stochastic/src/euler/philox.rs"]
mod philox;
mod program;
mod vocabulary;

#[macro_use]
mod entrypoint;
mod entries;

pub use args::EulerArgs;
pub(crate) use draws::normal;
pub use entries::*;
pub use kat::{cos_kat, log_kat, philox_kat, stable_kat};
pub(crate) use philox::{SITE_FGN, SITE_SHEET, SITE_SHEET_CORRECTION, philox4x32_10};
