use super::*;

#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyCompoundPoisson {
  inner_f32: Option<CompoundPoisson<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<CompoundPoisson<f64, crate::traits::CallableDist<f64>>>,
}

#[pyo3::prelude::pymethods]
impl PyCompoundPoisson {
  #[new]
  #[pyo3(signature = (distribution, lambda_, n=None, t_max=None, dtype=None))]
  fn new(
    distribution: pyo3::Py<pyo3::PyAny>,
    lambda_: f64,
    n: Option<usize>,
    t_max: Option<f64>,
    dtype: Option<&str>,
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> pyo3::PyResult<Self> {
      let horizon = crate::python_params::horizon_from_kwargs(n, t_max)?;
      Ok(match dtype.unwrap_or("f64") {
        "f32" => Self {
          inner_f32: Some(CompoundPoisson::new(
            crate::traits::CallableDist::new(distribution),
            Poisson::new(
              lambda_ as f32,
              crate::python_params::horizon_f32(horizon),
              Unseeded,
            ),
            Unseeded,
          )),
          inner_f64: None,
        },
        _ => Self {
          inner_f32: None,
          inner_f64: Some(CompoundPoisson::new(
            crate::traits::CallableDist::new(distribution),
            Poisson::new(lambda_, horizon, Unseeded),
            Unseeded,
          )),
        },
      })
    })?
  }

  fn sample(&self, py: pyo3::Python<'_>) -> pyo3::PyResult<crate::process::PyTriple> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use crate::traits::ProcessExt;
      if let Some(ref inner) = self.inner_f64 {
        inner.sample().into_py_tuple(py)
      } else if let Some(ref inner) = self.inner_f32 {
        inner.sample().into_py_tuple(py)
      } else {
        unreachable!()
      }
    })
  }
}
