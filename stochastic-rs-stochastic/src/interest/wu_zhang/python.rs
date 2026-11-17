use super::*;

use stochastic_rs_core::simd_rng::Deterministic;

#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyWuZhangD {
  inner_f32: Option<WuZhang<f32>>,
  inner_f64: Option<WuZhang<f64>>,
  seeded_f32: Option<WuZhang<f32, crate::simd_rng::Deterministic>>,
  seeded_f64: Option<WuZhang<f64, crate::simd_rng::Deterministic>>,
}

#[pyo3::prelude::pymethods]
impl PyWuZhangD {
  #[new]
  #[pyo3(signature = (alpha, beta, nu, lambda_, x0, v0, xn, n, t=None, seed=None, dtype=None))]
  fn new(
    alpha: Vec<f64>,
    beta: Vec<f64>,
    nu: Vec<f64>,
    lambda_: Vec<f64>,
    x0: Vec<f64>,
    v0: Vec<f64>,
    xn: usize,
    n: usize,
    t: Option<f64>,
    seed: Option<u64>,
    dtype: Option<&str>,
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| {
      match (seed, dtype.unwrap_or("f64")) {
        (Some(s), "f32") => {
          let to_f32_arr =
            |v: Vec<f64>| ndarray::Array1::from_vec(v.iter().map(|&x| x as f32).collect());
          Self {
            inner_f32: None,
            inner_f64: None,
            seeded_f32: Some(WuZhang::new(
              to_f32_arr(alpha),
              to_f32_arr(beta),
              to_f32_arr(nu),
              to_f32_arr(lambda_),
              to_f32_arr(x0),
              to_f32_arr(v0),
              xn,
              t.map(|v| v as f32),
              n,
              Deterministic::new(s),
            )),
            seeded_f64: None,
          }
        }
        (Some(s), _) => {
          let to_arr = |v: Vec<f64>| ndarray::Array1::from_vec(v);
          Self {
            inner_f32: None,
            inner_f64: None,
            seeded_f32: None,
            seeded_f64: Some(WuZhang::new(
              to_arr(alpha),
              to_arr(beta),
              to_arr(nu),
              to_arr(lambda_),
              to_arr(x0),
              to_arr(v0),
              xn,
              t,
              n,
              Deterministic::new(s),
            )),
          }
        }
        (None, "f32") => {
          let to_f32_arr =
            |v: Vec<f64>| ndarray::Array1::from_vec(v.iter().map(|&x| x as f32).collect());
          Self {
            inner_f32: Some(WuZhang::new(
              to_f32_arr(alpha),
              to_f32_arr(beta),
              to_f32_arr(nu),
              to_f32_arr(lambda_),
              to_f32_arr(x0),
              to_f32_arr(v0),
              xn,
              t.map(|v| v as f32),
              n,
              Unseeded,
            )),
            inner_f64: None,
            seeded_f32: None,
            seeded_f64: None,
          }
        }
        (None, _) => {
          let to_arr = |v: Vec<f64>| ndarray::Array1::from_vec(v);
          Self {
            inner_f32: None,
            inner_f64: Some(WuZhang::new(
              to_arr(alpha),
              to_arr(beta),
              to_arr(nu),
              to_arr(lambda_),
              to_arr(x0),
              to_arr(v0),
              xn,
              t,
              n,
              Unseeded,
            )),
            seeded_f32: None,
            seeded_f64: None,
          }
        }
      }
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

  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_dispatch!(self, |inner| {
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
