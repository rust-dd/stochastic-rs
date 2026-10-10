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
    #[doc(hidden)]
    #[pyo3::prelude::pyclass]
    pub struct $py_name {
      inner_f32: Option<std::sync::Mutex<$crate::Seeded<$inner<f32>>>>,
      inner_f64: Option<std::sync::Mutex<$crate::Seeded<$inner<f64>>>>,
    }

    #[pyo3::prelude::pymethods]
    impl $py_name {
      #[new]
      #[pyo3(signature = ($($sig)*))]
      fn new($($param: $pty,)* seed: Option<u64>, dtype: Option<&str>) -> pyo3::PyResult<Self> {
        $crate::python::value_error_on_panic(|| {
          use $crate::SimdDistribution;
          match (seed, dtype.unwrap_or("f64")) {
            (Some(sd), "f32") => Self {
              inner_f32: Some(std::sync::Mutex::new(
                $inner::new($(stochastic_rs_core::python::IntoF32::into_f32($param),)*)
                  .seeded(&stochastic_rs_core::simd_rng::Deterministic::new(sd)),
              )),
              inner_f64: None,
            },
            (Some(sd), _) => Self {
              inner_f32: None,
              inner_f64: Some(std::sync::Mutex::new(
                $inner::new($(stochastic_rs_core::python::IntoF64::into_f64($param),)*)
                  .seeded(&stochastic_rs_core::simd_rng::Deterministic::new(sd)),
              )),
            },
            (None, "f32") => Self {
              inner_f32: Some(std::sync::Mutex::new(
                $inner::new($(stochastic_rs_core::python::IntoF32::into_f32($param),)*)
                  .seeded(&stochastic_rs_core::simd_rng::Unseeded),
              )),
              inner_f64: None,
            },
            (None, _) => Self {
              inner_f32: None,
              inner_f64: Some(std::sync::Mutex::new(
                $inner::new($(stochastic_rs_core::python::IntoF64::into_f64($param),)*)
                  .seeded(&stochastic_rs_core::simd_rng::Unseeded),
              )),
            },
          }
        })
      }

      fn sample<'py>(&self, py: pyo3::Python<'py>, n: usize) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
        $crate::python::runtime_error_on_panic(|| {
          use $crate::DistributionSampler;
          use numpy::IntoPyArray;
          use pyo3::IntoPyObjectExt;
          if let Some(ref inner) = self.inner_f64 {
            py.detach(|| $crate::python::lock_stream(inner).sample_n(n)).into_pyarray(py).into_py_any(py).unwrap()
          } else if let Some(ref inner) = self.inner_f32 {
            py.detach(|| $crate::python::lock_stream(inner).sample_n(n)).into_pyarray(py).into_py_any(py).unwrap()
          } else {
            unreachable!()
          }
        })
      }

      fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize, n: usize) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
        $crate::python::runtime_error_on_panic(|| {
          use $crate::DistributionSampler;
          use numpy::IntoPyArray;
          use pyo3::IntoPyObjectExt;
          if let Some(ref inner) = self.inner_f64 {
            py.detach(|| $crate::python::lock_stream(inner).sample_matrix(m, n)).into_pyarray(py).into_py_any(py).unwrap()
          } else if let Some(ref inner) = self.inner_f32 {
            py.detach(|| $crate::python::lock_stream(inner).sample_matrix(m, n)).into_pyarray(py).into_py_any(py).unwrap()
          } else {
            unreachable!()
          }
        })
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
    #[doc(hidden)]
    #[pyo3::prelude::pyclass]
    pub struct $py_name {
      inner: std::sync::Mutex<$crate::Seeded<$inner<i64>>>,
    }

    #[pyo3::prelude::pymethods]
    impl $py_name {
      #[new]
      #[pyo3(signature = ($($sig)*))]
      fn new($($param: $pty,)* seed: Option<u64>) -> pyo3::PyResult<Self> {
        $crate::python::value_error_on_panic(|| {
          use $crate::SimdDistribution;
          match seed {
            Some(sd) => Self {
              inner: std::sync::Mutex::new(
                $inner::new($($param,)*)
                  .seeded(&stochastic_rs_core::simd_rng::Deterministic::new(sd)),
              ),
            },
            None => Self {
              inner: std::sync::Mutex::new(
                $inner::new($($param,)*).seeded(&stochastic_rs_core::simd_rng::Unseeded),
              ),
            },
          }
        })
      }

      fn sample<'py>(&self, py: pyo3::Python<'py>, n: usize) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
        $crate::python::runtime_error_on_panic(|| {
          use $crate::DistributionSampler;
          use numpy::IntoPyArray;
          use pyo3::IntoPyObjectExt;
          py.detach(|| $crate::python::lock_stream(&self.inner).sample_n(n)).into_pyarray(py).into_py_any(py).unwrap()
        })
      }

      fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize, n: usize) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
        $crate::python::runtime_error_on_panic(|| {
          use $crate::DistributionSampler;
          use numpy::IntoPyArray;
          use pyo3::IntoPyObjectExt;
          py.detach(|| $crate::python::lock_stream(&self.inner).sample_matrix(m, n)).into_pyarray(py).into_py_any(py).unwrap()
        })
      }
    }
  };
}

#[cfg(not(feature = "python"))]
#[macro_export]
macro_rules! py_distribution_int {
  ($($tt:tt)*) => {};
}
