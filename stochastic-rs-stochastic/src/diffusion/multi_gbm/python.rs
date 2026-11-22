use super::*;

#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyMultiGbm {
  inner: Option<MultiGbm<f64>>,
  seeded: Option<MultiGbm<f64, stochastic_rs_core::simd_rng::Deterministic>>,
}

#[pyo3::prelude::pymethods]
impl PyMultiGbm {
  /// Correlated multi-asset GBM: `mu`, `sigma`, `x0` per asset, `rho` the
  /// `k × k` correlation matrix, `n` grid points over `[0, t]`.
  #[new]
  #[pyo3(signature = (mu, sigma, rho, n, x0, t=None, seed=None))]
  fn new(
    mu: Vec<f64>,
    sigma: Vec<f64>,
    rho: numpy::PyReadonlyArray2<'_, f64>,
    n: usize,
    x0: Vec<f64>,
    t: Option<f64>,
    seed: Option<u64>,
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| {
      let rho = rho.as_array().to_owned();
      match seed {
        Some(s) => Self {
          inner: None,
          seeded: Some(MultiGbm::new(
            Array1::from_vec(mu),
            Array1::from_vec(sigma),
            rho,
            n,
            Array1::from_vec(x0),
            t,
            stochastic_rs_core::simd_rng::Deterministic::new(s),
          )),
        },
        None => Self {
          inner: Some(MultiGbm::new(
            Array1::from_vec(mu),
            Array1::from_vec(sigma),
            rho,
            n,
            Array1::from_vec(x0),
            t,
            Unseeded,
          )),
          seeded: None,
        },
      }
    })
  }

  /// One `(k, n)` path matrix.
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

  /// `m` independent path matrices as a list of `(k, n)` arrays.
  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;
      py_dispatch_f64!(self, |inner| {
        let samples = inner.sample_par(m);
        pyo3::types::PyList::new(
          py,
          samples
            .iter()
            .map(|s| s.clone().into_pyarray(py).into_py_any(py).unwrap()),
        )
        .unwrap()
        .into_py_any(py)
        .unwrap()
      })
    })
  }
}
