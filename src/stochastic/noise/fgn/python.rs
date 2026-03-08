//! # Python
//!
//! $$
//! \varepsilon \sim \mathcal N(0,\Sigma)\ \text{with optional fractional covariance shaping}
//! $$
//!
use numpy::IntoPyArray;
use numpy::PyArray1;
use numpy::PyArray2;
use numpy::ndarray::Array2;
use pyo3::prelude::*;

use super::FGN;
use crate::traits::ProcessExt;

#[pyclass]
pub struct PyFGN {
  inner: Option<FGN<f64>>,
  seeded: Option<FGN<f64, crate::simd_rng::Deterministic>>,
}

#[pymethods]
impl PyFGN {
  #[new]
  #[pyo3(signature = (hurst, n, t=None, seed=None))]
  fn new(hurst: f64, n: usize, t: Option<f64>, seed: Option<u64>) -> Self {
    match seed {
      Some(s) => Self {
        inner: None,
        seeded: Some(FGN::seeded(hurst, n, t, s)),
      },
      None => Self {
        inner: Some(FGN::new(hurst, n, t)),
        seeded: None,
      },
    }
  }

  fn sample<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    if let Some(ref inner) = self.inner {
      inner.sample().into_pyarray(py)
    } else if let Some(ref inner) = self.seeded {
      inner.sample().into_pyarray(py)
    } else {
      unreachable!()
    }
  }

  fn sample_par<'py>(&self, py: Python<'py>, m: usize) -> Bound<'py, PyArray2<f64>> {
    if let Some(ref inner) = self.inner {
      let paths = inner.sample_par(m);
      let n = paths[0].len();
      let mut result = Array2::<f64>::zeros((m, n));
      for (i, path) in paths.iter().enumerate() {
        result.row_mut(i).assign(path);
      }
      result.into_pyarray(py)
    } else if let Some(ref inner) = self.seeded {
      let paths = inner.sample_par(m);
      let n = paths[0].len();
      let mut result = Array2::<f64>::zeros((m, n));
      for (i, path) in paths.iter().enumerate() {
        result.row_mut(i).assign(path);
      }
      result.into_pyarray(py)
    } else {
      unreachable!()
    }
  }
}
