//! The drift and diffusion families of the Euler engine, declared once for host, CUDA and Metal.
//!
//! The DSL, a subset valid as Rust and as C, is specified in the `new-process` skill.

#[cfg(test)]
use crate::traits::FloatExt;

#[macro_use]
pub(crate) mod codegen;
pub(crate) mod vocabulary;

#[cfg(test)]
pub(crate) use vocabulary::ops;

#[macro_use]
mod definitions;

euler_family_definitions!(euler_families);

#[cfg(test)]
mod tests;
