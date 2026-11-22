//! Python-boundary helpers for the binding crates: PyO3 raises a stopped panic as `PanicException`,
//! a `BaseException` that `except Exception` misses, so these re-raise it by cause.

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

/// Locks a stream; call it in `Python::detach` only, so no caller waits holding the interpreter; a
/// panic caught by [`runtime_error_on_panic`] leaves the stream valid, and poison cleared.
pub fn lock_stream<T>(stream: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
  stream
    .lock()
    .unwrap_or_else(std::sync::PoisonError::into_inner)
}
