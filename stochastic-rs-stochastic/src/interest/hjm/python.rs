use super::*;

use stochastic_rs_core::simd_rng::Deterministic;

#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyHjm {
  inner: Option<Hjm<f64>>,
  seeded: Option<Hjm<f64, stochastic_rs_core::simd_rng::Deterministic>>,
}

#[pyo3::prelude::pymethods]
impl PyHjm {
  #[new]
  #[pyo3(signature = (a, b, p, q, v, alpha, sigma, n, r0=None, p0=None, f0=None, t=None, seed=None))]
  fn new(
    a: pyo3::Py<pyo3::PyAny>,
    b: pyo3::Py<pyo3::PyAny>,
    p: pyo3::Py<pyo3::PyAny>,
    q: pyo3::Py<pyo3::PyAny>,
    v: pyo3::Py<pyo3::PyAny>,
    alpha: pyo3::Py<pyo3::PyAny>,
    sigma: pyo3::Py<pyo3::PyAny>,
    n: usize,
    r0: Option<f64>,
    p0: Option<f64>,
    f0: Option<f64>,
    t: Option<f64>,
    seed: Option<u64>,
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| {
      use crate::traits::Fn2D;
      match seed {
        Some(s) => Self {
          inner: None,
          seeded: Some(Hjm::new(
            Fn1D::Py(a),
            Fn1D::Py(b),
            Fn2D::Py(p),
            Fn2D::Py(q),
            Fn2D::Py(v),
            Fn2D::Py(alpha),
            Fn2D::Py(sigma),
            n,
            r0,
            p0,
            f0,
            t,
            Deterministic::new(s),
          )),
        },
        None => Self {
          inner: Some(Hjm::new(
            Fn1D::Py(a),
            Fn1D::Py(b),
            Fn2D::Py(p),
            Fn2D::Py(q),
            Fn2D::Py(v),
            Fn2D::Py(alpha),
            Fn2D::Py(sigma),
            n,
            r0,
            p0,
            f0,
            t,
            Unseeded,
          )),
          seeded: None,
        },
      }
    })
  }

  fn sample<'py>(
    &self,
    py: pyo3::Python<'py>,
  ) -> pyo3::PyResult<(
    pyo3::Py<pyo3::PyAny>,
    pyo3::Py<pyo3::PyAny>,
    pyo3::Py<pyo3::PyAny>,
  )> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_dispatch_f64!(self, |inner| {
        let HjmPaths { r, p, f, .. } = inner.sample();
        (
          r.into_pyarray(py).into_py_any(py).unwrap(),
          p.into_pyarray(py).into_py_any(py).unwrap(),
          f.into_pyarray(py).into_py_any(py).unwrap(),
        )
      })
    })
  }

  /// `m` independent paths as three `(m, n)` arrays: short rate, bond price and forward rate.
  /// The GIL is released meanwhile; the Python callables re-acquire it per evaluation.
  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> pyo3::PyResult<(
    pyo3::Py<pyo3::PyAny>,
    pyo3::Py<pyo3::PyAny>,
    pyo3::Py<pyo3::PyAny>,
  )> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use ndarray::Array2;
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_dispatch_f64!(self, |inner| {
        let paths = py.detach(|| inner.sample_par(m));
        let n = paths.first().map_or(0, |p| p.r.len());
        let mut stacked = [
          Array2::zeros((m, n)),
          Array2::zeros((m, n)),
          Array2::zeros((m, n)),
        ];
        for (i, HjmPaths { r, p, f, .. }) in paths.iter().enumerate() {
          stacked[0].row_mut(i).assign(r);
          stacked[1].row_mut(i).assign(p);
          stacked[2].row_mut(i).assign(f);
        }
        let [r, p, f] = stacked;
        (
          r.into_pyarray(py).into_py_any(py).unwrap(),
          p.into_pyarray(py).into_py_any(py).unwrap(),
          f.into_pyarray(py).into_py_any(py).unwrap(),
        )
      })
    })
  }
}
