//! Macro shims.
//!
//! The actual Python wrapper code generation lives in `stochastic-rs-py`;
//! these no-op definitions exist so that macro invocations inline in the
//! distribution source files compile cleanly under the workspace layout.
//! When the umbrella crate is compiled with `feature = "python"`, the
//! `stochastic-rs-py` crate re-creates the bindings from scratch.

#[macro_export]
macro_rules! py_distribution {
  ($($tt:tt)*) => {};
}

#[macro_export]
macro_rules! py_distribution_int {
  ($($tt:tt)*) => {};
}
