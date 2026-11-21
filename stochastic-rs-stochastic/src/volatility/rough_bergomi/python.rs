use super::*;

use stochastic_rs_core::simd_rng::Deterministic;

#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyRoughBergomi {
  inner_f32: Option<RoughBergomi<f32>>,
  inner_f64: Option<RoughBergomi<f64>>,
  seeded_f32: Option<RoughBergomi<f32, stochastic_rs_core::simd_rng::Deterministic>>,
  seeded_f64: Option<RoughBergomi<f64, stochastic_rs_core::simd_rng::Deterministic>>,
}

#[pyo3::prelude::pymethods]
impl PyRoughBergomi {
  #[new]
  #[pyo3(signature = (hurst, nu, r, rho, n, v0, s0, t=None, seed=None, dtype=None))]
  fn new(
    hurst: f64,
    nu: f64,
    r: f64,
    rho: f64,
    n: usize,
    v0: f64,
    s0: f64,
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
      match (seed, dtype.unwrap_or("f64")) {
        (Some(sd), "f32") => {
          s.seeded_f32 = Some(RoughBergomi::new(
            hurst as f32,
            nu as f32,
            v0 as f32,
            s0 as f32,
            r as f32,
            rho as f32,
            n,
            t.map(|v| v as f32),
            Deterministic::new(sd),
          ));
        }
        (Some(sd), _) => {
          s.seeded_f64 = Some(RoughBergomi::new(
            hurst,
            nu,
            v0,
            s0,
            r,
            rho,
            n,
            t,
            Deterministic::new(sd),
          ));
        }
        (None, "f32") => {
          s.inner_f32 = Some(RoughBergomi::new(
            hurst as f32,
            nu as f32,
            v0 as f32,
            s0 as f32,
            r as f32,
            rho as f32,
            n,
            t.map(|v| v as f32),
            Unseeded,
          ));
        }
        (None, _) => {
          s.inner_f64 = Some(RoughBergomi::new(hurst, nu, v0, s0, r, rho, n, t, Unseeded));
        }
      }
      s
    })
  }

  fn sample<'py>(
    &self,
    py: pyo3::Python<'py>,
  ) -> pyo3::PyResult<(pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>)> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_dispatch!(self, |inner| {
        let [a, b] = inner.sample();
        (
          a.into_pyarray(py).into_py_any(py).unwrap(),
          b.into_pyarray(py).into_py_any(py).unwrap(),
        )
      })
    })
  }

  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> pyo3::PyResult<(pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>)> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use numpy::ndarray::Array2;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_dispatch!(self, |inner| {
        let samples = inner.sample_par(m);
        let n = samples[0][0].len();
        let mut r0 = Array2::zeros((m, n));
        let mut r1 = Array2::zeros((m, n));
        for (i, [a, b]) in samples.iter().enumerate() {
          r0.row_mut(i).assign(a);
          r1.row_mut(i).assign(b);
        }
        (
          r0.into_pyarray(py).into_py_any(py).unwrap(),
          r1.into_pyarray(py).into_py_any(py).unwrap(),
        )
      })
    })
  }
}
