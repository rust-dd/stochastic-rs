//! # Process
//!
//! Brownian, fractional and stable motions, point processes and subordinators.
//!
pub mod bm;
pub mod brownian_bridge;
pub mod cbms;
pub mod ccustom;
pub mod cfbms;
pub mod cpoisson;
pub mod customjt;
pub mod fbm;
pub mod hawkes;
pub mod lfsm;
pub mod multivariate_hawkes;
pub mod poisson;
pub mod subordinator;

use ndarray::Array1;

/// Arrival times, the running sum of the jumps and the jump sizes.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct CompoundPaths<T> {
  pub arrivals: Array1<T>,
  pub cumulative: Array1<T>,
  pub jumps: Array1<T>,
}

impl<T> From<[Array1<T>; 3]> for CompoundPaths<T> {
  fn from([arrivals, cumulative, jumps]: [Array1<T>; 3]) -> Self {
    Self {
      arrivals,
      cumulative,
      jumps,
    }
  }
}

#[cfg(feature = "python")]
pub(crate) type PyTriple = (
  pyo3::Py<pyo3::PyAny>,
  pyo3::Py<pyo3::PyAny>,
  pyo3::Py<pyo3::PyAny>,
);

#[cfg(feature = "python")]
impl<T: numpy::Element> CompoundPaths<T> {
  /// The tuple `(arrivals, cumulative, jumps)` the Python classes return.
  pub(crate) fn into_py_tuple(self, py: pyo3::Python<'_>) -> PyTriple {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    (
      self.arrivals.into_pyarray(py).into_py_any(py).unwrap(),
      self.cumulative.into_pyarray(py).into_py_any(py).unwrap(),
      self.jumps.into_pyarray(py).into_py_any(py).unwrap(),
    )
  }
}
