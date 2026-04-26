//! Macro shims.
//!
//! Real Python wrapper code generation lives in `stochastic-rs-py`. The empty
//! macros below let inline `py_process_*!` invocations compile as no-ops at
//! the sub-crate level.

#[macro_export]
macro_rules! py_process_1d {
  ($($tt:tt)*) => {};
}

#[macro_export]
macro_rules! py_process_2x1d {
  ($($tt:tt)*) => {};
}

#[macro_export]
macro_rules! py_process_2d {
  ($($tt:tt)*) => {};
}
