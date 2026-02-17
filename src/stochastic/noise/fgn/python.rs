//! # Python
//!
//! $$
//! \varepsilon \sim \mathcal N(0,\Sigma)\ \text{with optional fractional covariance shaping}
//! $$
//!
use numpy::ndarray::Array2;
use numpy::IntoPyArray;
use numpy::PyArray1;
use numpy::PyArray2;
use pyo3::prelude::*;

use super::FGN;
use crate::traits::ProcessExt;

#[pyclass]
pub struct PyFGN {
  inner: FGN<f64>,
}

#[pymethods]
impl PyFGN {
  #[new]
  #[pyo3(signature = (hurst, n, t=None))]
  fn new(hurst: f64, n: usize, t: Option<f64>) -> Self {
    Self {
      inner: FGN::new(hurst, n, t),
    }
  }

  fn sample<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    self.inner.sample().into_pyarray(py)
  }

  fn sample_par<'py>(&self, py: Python<'py>, m: usize) -> Bound<'py, PyArray2<f64>> {
    let paths = self.inner.sample_par(m);
    let n = paths[0].len();
    let mut result = Array2::<f64>::zeros((m, n));
    for (i, path) in paths.iter().enumerate() {
      result.row_mut(i).assign(path);
    }
    result.into_pyarray(py)
  }
}