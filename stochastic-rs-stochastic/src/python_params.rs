//! The Python kwargs of [`crate::params`] and their `None` defaults, mapped by the class macros.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::params::VarianceBoundary;

/// `"floor"` or `"reflect"`; any other value is a `ValueError`.
pub struct PyVarianceBoundary(pub VarianceBoundary);

impl<'a, 'py> FromPyObject<'a, 'py> for PyVarianceBoundary {
  type Error = PyErr;

  fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
    match ob.extract::<String>().as_deref() {
      Ok("floor") => Ok(Self(VarianceBoundary::Floor)),
      Ok("reflect") => Ok(Self(VarianceBoundary::Reflect)),
      _ => Err(PyValueError::new_err(format!(
        "boundary must be \"floor\" or \"reflect\", got {}",
        ob.repr()?
      ))),
    }
  }
}

/// `boundary=None` is the floor.
pub fn boundary(boundary: Option<PyVarianceBoundary>) -> VarianceBoundary {
  boundary.map_or(VarianceBoundary::Floor, |b| b.0)
}

/// `clamp=None` clamps, as the Rust default does.
pub fn clamp(clamp: Option<bool>) -> bool {
  clamp.unwrap_or(true)
}
