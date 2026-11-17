use super::*;

#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyWishart {
  inner: Option<Wishart<f64>>,
  seeded: Option<Wishart<f64, stochastic_rs_core::simd_rng::Deterministic>>,
}

#[pyo3::prelude::pymethods]
impl PyWishart {
  /// Wishart process with degree `alpha`, drift `b`, volatility `a` and
  /// initial matrix `x0` (all `d × d` arrays), `n` grid points over `[0, t]`.
  #[new]
  #[pyo3(signature = (alpha, b, a, x0, n, t=None, seed=None))]
  fn new(
    alpha: f64,
    b: numpy::PyReadonlyArray2<'_, f64>,
    a: numpy::PyReadonlyArray2<'_, f64>,
    x0: numpy::PyReadonlyArray2<'_, f64>,
    n: usize,
    t: Option<f64>,
    seed: Option<u64>,
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| {
      let (b, a, x0) = (
        b.as_array().to_owned(),
        a.as_array().to_owned(),
        x0.as_array().to_owned(),
      );
      match seed {
        Some(s) => Self {
          inner: None,
          seeded: Some(Wishart::new(
            alpha,
            b,
            a,
            x0,
            n,
            t,
            stochastic_rs_core::simd_rng::Deterministic::new(s),
          )),
        },
        None => Self {
          inner: Some(Wishart::new(alpha, b, a, x0, n, t, Unseeded)),
          seeded: None,
        },
      }
    })
  }

  /// One `(n, d, d)` array of matrices along the grid.
  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;
      py_dispatch_f64!(self, |inner| inner
        .sample()
        .into_pyarray(py)
        .into_py_any(py)
        .unwrap())
    })
  }

  /// `m` independent paths as a list of `(n, d, d)` arrays.
  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;
      py_dispatch_f64!(self, |inner| {
        let paths: Vec<pyo3::Py<pyo3::PyAny>> = inner
          .sample_par(m)
          .into_iter()
          .map(|p| p.into_pyarray(py).into_py_any(py).unwrap())
          .collect();
        paths.into_py_any(py).unwrap()
      })
    })
  }

  /// Closed-form mean `E[X_t]` as a `(d, d)` array.
  fn mean<'py>(&self, py: pyo3::Python<'py>, t: f64) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;
    py_dispatch_f64!(self, |inner| inner
      .mean(t)
      .into_pyarray(py)
      .into_py_any(py)
      .unwrap())
  }

  /// Laplace transform `E[exp(Tr(v X_t))]` for a symmetric `v` in the
  /// convergence domain (every negative semidefinite `v` qualifies).
  fn laplace_transform(&self, v: numpy::PyReadonlyArray2<'_, f64>, t: f64) -> f64 {
    let v = v.as_array().to_owned();
    py_dispatch_f64!(self, |inner| inner.laplace_transform(&v, t))
  }
}
