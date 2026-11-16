use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use super::stats_error;

#[pyclass(name = "GaussianKDE", unsendable)]
pub struct PyGaussianKDE {
  inner: crate::kde::GaussianKde,
}

#[pymethods]
impl PyGaussianKDE {
  /// Construct a Gaussian KDE with explicit bandwidth.
  #[new]
  fn new<'py>(data: PyReadonlyArray1<'py, f64>, bandwidth: f64) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> PyResult<Self> {
      Ok(Self {
        inner: crate::kde::GaussianKde::new(data.as_array(), bandwidth).map_err(stats_error)?,
      })
    })?
  }

  /// Construct a Gaussian KDE with Silverman's rule-of-thumb bandwidth.
  #[staticmethod]
  fn silverman<'py>(data: PyReadonlyArray1<'py, f64>) -> PyResult<Self> {
    crate::kde::GaussianKde::with_silverman_bandwidth(data.as_array())
      .map(|inner| Self { inner })
      .map_err(stats_error)
  }

  fn evaluate(&self, x: f64) -> f64 {
    self.inner.evaluate(x)
  }

  fn evaluate_array<'py>(
    &self,
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
  ) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    let out = self.inner.evaluate_array(x.as_array());
    out.into_pyarray(py)
  }
}
