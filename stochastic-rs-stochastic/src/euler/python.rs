//! Python surface of the device layer: probing a device and choosing the ordinal.
//!
//! Sampling on a device goes through the process classes' `device=` argument.

use pyo3::prelude::*;

/// Describes the named device (`"cpu"`, `"accelerate"`, `"cuda"`, `"metal"`, optionally
/// `:ordinal`) as a dict; `RuntimeError` if it cannot be used, `ValueError` if the build lacks it.
#[pyfunction]
pub fn probe_device<'py>(
  py: Python<'py>,
  device: &str,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
  let info = crate::python_device::Device::parse_name(device)?.probe()?;
  let d = pyo3::types::PyDict::new(py);
  d.set_item("backend", info.backend)?;
  d.set_item("name", info.name)?;
  d.set_item("precisions", info.precisions.to_vec())?;
  d.set_item("ordinal", info.ordinal)?;
  Ok(d)
}
