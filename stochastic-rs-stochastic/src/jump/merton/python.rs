use super::*;

use stochastic_rs_core::simd_rng::Deterministic;

#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyMerton {
  inner_f32: Option<Merton<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<Merton<f64, crate::traits::CallableDist<f64>>>,
  seeded_f32: Option<Merton<f32, crate::traits::CallableDist<f32>, crate::simd_rng::Deterministic>>,
  seeded_f64: Option<Merton<f64, crate::traits::CallableDist<f64>, crate::simd_rng::Deterministic>>,
}

#[pyo3::prelude::pymethods]
impl PyMerton {
  #[new]
  #[pyo3(signature = (alpha, sigma, lambda_, theta, distribution, n, x0=None, t=None, seed=None, dtype=None))]
  fn new(
    alpha: f64,
    sigma: f64,
    lambda_: f64,
    theta: f64,
    distribution: pyo3::Py<pyo3::PyAny>,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    seed: Option<u64>,
    dtype: Option<&str>,
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| {
      let mut s = Self {
        inner_f32: None,
        inner_f64: None,
        seeded_f32: None,
        seeded_f64: None,
      };
      match dtype.unwrap_or("f64") {
        "f32" => {
          let jump_dist = crate::traits::CallableDist::new(distribution);
          match seed {
            Some(sd) => {
              s.seeded_f32 = Some(Merton::new(
                alpha as f32,
                sigma as f32,
                lambda_ as f32,
                theta as f32,
                jump_dist,
                n,
                x0.map(|v| v as f32),
                t.map(|v| v as f32),
                Deterministic::new(sd),
              ));
            }
            None => {
              s.inner_f32 = Some(Merton::new(
                alpha as f32,
                sigma as f32,
                lambda_ as f32,
                theta as f32,
                jump_dist,
                n,
                x0.map(|v| v as f32),
                t.map(|v| v as f32),
                Unseeded,
              ));
            }
          }
        }
        _ => {
          let jump_dist = crate::traits::CallableDist::new(distribution);
          match seed {
            Some(sd) => {
              s.seeded_f64 = Some(Merton::new(
                alpha,
                sigma,
                lambda_,
                theta,
                jump_dist,
                n,
                x0,
                t,
                Deterministic::new(sd),
              ));
            }
            None => {
              s.inner_f64 = Some(Merton::new(
                alpha, sigma, lambda_, theta, jump_dist, n, x0, t, Unseeded,
              ));
            }
          }
        }
      }
      s
    })
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_dispatch!(self, |inner| inner
        .sample()
        .into_pyarray(py)
        .into_py_any(py)
        .unwrap())
    })
  }

  /// `m` paths as an `(m, n)` array. The GIL is released while they are generated; the Python
  /// law runs on rayon workers, one call at a time.
  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use numpy::ndarray::Array2;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_dispatch!(self, |inner| {
        let paths = py.detach(|| inner.sample_par(m));
        let n = paths.first().map_or(0, |p| p.len());
        let mut result = Array2::zeros((m, n));
        for (i, path) in paths.iter().enumerate() {
          result.row_mut(i).assign(path);
        }
        result.into_pyarray(py).into_py_any(py).unwrap()
      })
    })
  }
}
