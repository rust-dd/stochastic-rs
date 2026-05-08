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
      is_seeded: bool,
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
            is_seeded: true,
          },
          (Some(sd), _) => Self {
            inner_f32: None,
            inner_f64: Some($inner::from_seed_source(
              $(stochastic_rs_core::python::IntoF64::into_f64($param),)*
              &stochastic_rs_core::simd_rng::Deterministic::new(sd),
            )),
            is_seeded: true,
          },
          (None, "f32") => Self {
            inner_f32: Some($inner::new($(stochastic_rs_core::python::IntoF32::into_f32($param)),*)),
            inner_f64: None,
            is_seeded: false,
          },
          (None, _) => Self {
            inner_f32: None,
            inner_f64: Some($inner::new($(stochastic_rs_core::python::IntoF64::into_f64($param)),*)),
            is_seeded: false,
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
        use numpy::ndarray::Array2;
        use pyo3::IntoPyObjectExt;
        // Seeded path must serialize: `sample_matrix`'s rayon worker pool
        // clones `self`, and `Clone for Simd*` resets the rng, so the
        // seeded stream would be lost. The serial `fill_slice` path keeps
        // using the internal rng (every Simd* `fill_slice<R>` ignores the
        // passed rng and uses the seeded `self.simd_rng`).
        if self.is_seeded {
          if let Some(ref inner) = self.inner_f64 {
            let mut buf = Array2::<f64>::zeros((m, n));
            let mut dummy = stochastic_rs_core::simd_rng::SimdRng::from_seed(0);
            inner.fill_slice(&mut dummy, buf.as_slice_mut().unwrap());
            buf.into_pyarray(py).into_py_any(py).unwrap()
          } else if let Some(ref inner) = self.inner_f32 {
            let mut buf = Array2::<f32>::zeros((m, n));
            let mut dummy = stochastic_rs_core::simd_rng::SimdRng::from_seed(0);
            inner.fill_slice(&mut dummy, buf.as_slice_mut().unwrap());
            buf.into_pyarray(py).into_py_any(py).unwrap()
          } else {
            unreachable!()
          }
        } else if let Some(ref inner) = self.inner_f64 {
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
      is_seeded: bool,
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
            is_seeded: true,
          },
          None => Self {
            inner: $inner::new($($param),*),
            is_seeded: false,
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
        use numpy::ndarray::Array2;
        use pyo3::IntoPyObjectExt;
        // See `py_distribution!` for the seeded-path serialization rationale.
        if self.is_seeded {
          let mut buf = Array2::<i64>::zeros((m, n));
          let mut dummy = stochastic_rs_core::simd_rng::SimdRng::from_seed(0);
          self.inner.fill_slice(&mut dummy, buf.as_slice_mut().unwrap());
          buf.into_pyarray(py).into_py_any(py).unwrap()
        } else {
          self.inner.sample_matrix(m, n).into_pyarray(py).into_py_any(py).unwrap()
        }
      }
    }
  };
}

#[cfg(not(feature = "python"))]
#[macro_export]
macro_rules! py_distribution_int {
  ($($tt:tt)*) => {};
}
