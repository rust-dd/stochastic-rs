//! PyO3 wrapper macros for distribution types.
//!
//! Generate `PyXxx` newtype + `#[pymethods]` impl for each `Simd*` distribution
//! when the `python` feature is enabled. The `stochastic-rs-py` cdylib then
//! collects and registers them.

#[cfg(feature = "python")]
#[macro_export]
macro_rules! py_distribution {
  ($py_name:ident, $inner:ident,
    sig: ($($sig:tt)*),
    params: ($($param:ident : $pty:ty),* $(,)?)
  ) => {
    #[pyo3::prelude::pyclass(unsendable)]
    pub struct $py_name {
      inner_f32: Option<$inner<f32>>,
      inner_f64: Option<$inner<f64>>,
    }

    #[pyo3::prelude::pymethods]
    impl $py_name {
      #[new]
      #[pyo3(signature = ($($sig)*))]
      fn new($($param: $pty,)* seed: Option<u64>, dtype: Option<&str>) -> Self {
        match (seed, dtype.unwrap_or("f64")) {
          (Some(sd), "f32") => Self {
            inner_f32: Some($inner::from_seed_source(
              $(stochastic_rs_core::python::IntoF32::into_f32($param),)*
              &stochastic_rs_core::simd_rng::Deterministic::new(sd),
            )),
            inner_f64: None,
          },
          (Some(sd), _) => Self {
            inner_f32: None,
            inner_f64: Some($inner::from_seed_source(
              $(stochastic_rs_core::python::IntoF64::into_f64($param),)*
              &stochastic_rs_core::simd_rng::Deterministic::new(sd),
            )),
          },
          (None, "f32") => Self {
            inner_f32: Some($inner::new($(stochastic_rs_core::python::IntoF32::into_f32($param)),*)),
            inner_f64: None,
          },
          (None, _) => Self {
            inner_f32: None,
            inner_f64: Some($inner::new($(stochastic_rs_core::python::IntoF64::into_f64($param)),*)),
          },
        }
      }

      fn sample<'py>(&self, py: pyo3::Python<'py>, n: usize) -> pyo3::Py<pyo3::PyAny> {
        use $crate::DistributionSampler;
        use numpy::IntoPyArray;
        use pyo3::IntoPyObjectExt;
        if let Some(ref inner) = self.inner_f64 {
          inner.sample_n(n).into_pyarray(py).into_py_any(py).unwrap()
        } else if let Some(ref inner) = self.inner_f32 {
          inner.sample_n(n).into_pyarray(py).into_py_any(py).unwrap()
        } else {
          unreachable!()
        }
      }

      fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize, n: usize) -> pyo3::Py<pyo3::PyAny> {
        use $crate::DistributionSampler;
        use numpy::IntoPyArray;
        use pyo3::IntoPyObjectExt;
        if let Some(ref inner) = self.inner_f64 {
          inner.sample_matrix(m, n).into_pyarray(py).into_py_any(py).unwrap()
        } else if let Some(ref inner) = self.inner_f32 {
          inner.sample_matrix(m, n).into_pyarray(py).into_py_any(py).unwrap()
        } else {
          unreachable!()
        }
      }
    }
  };
}

#[cfg(not(feature = "python"))]
#[macro_export]
macro_rules! py_distribution {
  ($($tt:tt)*) => {};
}

#[cfg(feature = "python")]
#[macro_export]
macro_rules! py_distribution_int {
  ($py_name:ident, $inner:ident,
    sig: ($($sig:tt)*),
    params: ($($param:ident : $pty:ty),* $(,)?)
  ) => {
    #[pyo3::prelude::pyclass(unsendable)]
    pub struct $py_name {
      inner: $inner<i64>,
    }

    #[pyo3::prelude::pymethods]
    impl $py_name {
      #[new]
      #[pyo3(signature = ($($sig)*))]
      fn new($($param: $pty,)* seed: Option<u64>) -> Self {
        match seed {
          Some(sd) => Self {
            inner: $inner::from_seed_source(
              $($param,)*
              &stochastic_rs_core::simd_rng::Deterministic::new(sd),
            ),
          },
          None => Self {
            inner: $inner::new($($param),*),
          },
        }
      }

      fn sample<'py>(&self, py: pyo3::Python<'py>, n: usize) -> pyo3::Py<pyo3::PyAny> {
        use $crate::DistributionSampler;
        use numpy::IntoPyArray;
        use pyo3::IntoPyObjectExt;
        self.inner.sample_n(n).into_pyarray(py).into_py_any(py).unwrap()
      }

      fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize, n: usize) -> pyo3::Py<pyo3::PyAny> {
        use $crate::DistributionSampler;
        use numpy::IntoPyArray;
        use pyo3::IntoPyObjectExt;
        self.inner.sample_matrix(m, n).into_pyarray(py).into_py_any(py).unwrap()
      }
    }
  };
}

#[cfg(not(feature = "python"))]
#[macro_export]
macro_rules! py_distribution_int {
  ($($tt:tt)*) => {};
}
