//! Python surface of the device layer: probing a device and choosing the ordinal.
//!
//! Sampling on a device goes through the process classes' `device=` argument.

use pyo3::prelude::*;

/// Probes `cpu|accelerate|cuda|metal[:ordinal]` into a dict; `RuntimeError` if unusable,
/// `ValueError` for a bad name, `:ordinal` or `STOCHASTIC_RS_DEVICE_*`, or a build lacking it.
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
