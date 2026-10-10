//! Helpers every binding crate uses at the Python boundary.
//!
//! PyO3 already stops a Rust panic before it unwinds into the interpreter, but
//! it raises it as `pyo3_runtime.PanicException`, a `BaseException` that an
//! `except Exception` clause does not catch. These wrappers give the panic the
//! class a Python caller expects for its cause.

use std::any::Any;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

use pyo3::PyResult;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;

fn panic_message(payload: Box<dyn Any + Send>) -> String {
  payload
    .downcast_ref::<&str>()
    .map(|message| (*message).to_string())
    .or_else(|| payload.downcast_ref::<String>().cloned())
    .unwrap_or_else(|| "a Rust panic without a message".to_string())
}

/// Runs a constructor, where a panic is an assertion on an argument, and
/// reports one as a Python `ValueError` carrying the panic message.
pub fn value_error_on_panic<T>(f: impl FnOnce() -> T) -> PyResult<T> {
  catch_unwind(AssertUnwindSafe(f)).map_err(|payload| PyValueError::new_err(panic_message(payload)))
}

/// Runs a sampling call, where a panic is a failure of the run — a device
/// error, an exhausted resource — and reports one as a Python `RuntimeError`.
pub fn runtime_error_on_panic<T>(f: impl FnOnce() -> T) -> PyResult<T> {
  catch_unwind(AssertUnwindSafe(f))
    .map_err(|payload| PyRuntimeError::new_err(panic_message(payload)))
}

/// Locks a stream; call it only inside `Python::detach`, so no caller waits here holding the interpreter its holder
/// may need. A panic caught by [`runtime_error_on_panic`] leaves the stream valid, so a poisoned lock is recovered.
pub fn lock_stream<T>(stream: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
  stream
    .lock()
    .unwrap_or_else(std::sync::PoisonError::into_inner)
}
