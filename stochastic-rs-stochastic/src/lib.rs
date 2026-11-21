#![doc = include_str!("../README.md")]
// Defaults to `warn`, which is how 94 broken doc links accumulated
// unnoticed; deny so a regression fails the build instead of drifting.
#![deny(rustdoc::broken_intra_doc_links)]
#![allow(clippy::type_complexity)]
#![allow(
  clippy::too_many_arguments,
  reason = "a constructor takes its model's parameters by position, and a launch its buffers"
)]
// Numerical-loop indices are kept as-is; iterator chains hurt readability
// in matrix/path index code where i, j, k carry mathematical meaning.
#![allow(clippy::needless_range_loop)]

#[macro_use]
mod macros;
#[cfg(doctest)]
mod doctest_seals;
mod linalg;

pub mod traits;

pub use crate::device::Backend;
pub use crate::device::Cpu;
pub use crate::device::FgnBackend;
pub use crate::device::SheetBackend;
pub use crate::traits::ProcessExt;

pub mod autoregressive;
pub(crate) mod buffer;
pub mod correlation;
pub mod device;
pub mod diffusion;
pub mod euler;
pub mod glossary;
pub mod interest;
pub mod isonormal;
pub mod ito;
pub mod jump;
pub mod malliavin;
pub mod mc;
pub mod noise;
pub mod numerics;
pub mod params;
pub(crate) mod precondition;
pub mod process;
#[cfg(feature = "python")]
#[doc(hidden)]
pub mod python_device;
#[cfg(feature = "python")]
#[doc(hidden)]
pub mod python_params;
pub mod rough;
pub mod sde;
pub mod sheet;
pub mod volatility;
pub mod volterra;
