//! # Macros
//!
//! $$
//! \text{macro expansion} : (\text{model spec}) \mapsto (\text{impl blocks})
//! $$
//!
#[cfg(feature = "python")]
macro_rules! py_process_1d {
  ($py_name:ident, $inner:ident,
    sig: ($($sig:tt)*),
    params: ($($param:ident : $pty:ty),* $(,)?)
  ) => {
    #[pyo3::prelude::pyclass]
    pub struct $py_name {
      inner_f32: Option<$inner<f32>>,
      inner_f64: Option<$inner<f64>>,
    }

    #[pyo3::prelude::pymethods]
    impl $py_name {
      #[new]
      #[pyo3(signature = ($($sig)*))]
      fn new($($param: $pty,)* dtype: Option<&str>) -> Self {
        match dtype.unwrap_or("f64") {
          "f32" => Self {
            inner_f32: Some($inner::new($(crate::python::IntoF32::into_f32($param)),*)),
            inner_f64: None,
          },
          _ => Self {
            inner_f32: None,
            inner_f64: Some($inner::new($(crate::python::IntoF64::into_f64($param)),*)),
          },
        }
      }

      fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
        use numpy::IntoPyArray;
        use crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        if let Some(ref inner) = self.inner_f64 {
          inner.sample().into_pyarray(py).into_py_any(py).unwrap()
        } else if let Some(ref inner) = self.inner_f32 {
          inner.sample().into_pyarray(py).into_py_any(py).unwrap()
        } else {
          unreachable!()
        }
      }

      fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
        use numpy::IntoPyArray;
        use numpy::ndarray::Array2;
        use crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        if let Some(ref inner) = self.inner_f64 {
          let paths = inner.sample_par(m);
          let n = paths[0].len();
          let mut result = Array2::<f64>::zeros((m, n));
          for (i, path) in paths.iter().enumerate() {
            result.row_mut(i).assign(path);
          }
          result.into_pyarray(py).into_py_any(py).unwrap()
        } else if let Some(ref inner) = self.inner_f32 {
          let paths = inner.sample_par(m);
          let n = paths[0].len();
          let mut result = Array2::<f32>::zeros((m, n));
          for (i, path) in paths.iter().enumerate() {
            result.row_mut(i).assign(path);
          }
          result.into_pyarray(py).into_py_any(py).unwrap()
        } else {
          unreachable!()
        }
      }
    }
  };
}

#[cfg(not(feature = "python"))]
macro_rules! py_process_1d {
  ($($tt:tt)*) => {};
}

#[cfg(feature = "python")]
macro_rules! py_process_2x1d {
  ($py_name:ident, $inner:ident,
    sig: ($($sig:tt)*),
    params: ($($param:ident : $pty:ty),* $(,)?)
  ) => {
    #[pyo3::prelude::pyclass]
    pub struct $py_name {
      inner_f32: Option<$inner<f32>>,
      inner_f64: Option<$inner<f64>>,
    }

    #[pyo3::prelude::pymethods]
    impl $py_name {
      #[new]
      #[pyo3(signature = ($($sig)*))]
      fn new($($param: $pty,)* dtype: Option<&str>) -> Self {
        match dtype.unwrap_or("f64") {
          "f32" => Self {
            inner_f32: Some($inner::new($(crate::python::IntoF32::into_f32($param)),*)),
            inner_f64: None,
          },
          _ => Self {
            inner_f32: None,
            inner_f64: Some($inner::new($(crate::python::IntoF64::into_f64($param)),*)),
          },
        }
      }

      fn sample<'py>(&self, py: pyo3::Python<'py>) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
        use numpy::IntoPyArray;
        use crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        if let Some(ref inner) = self.inner_f64 {
          let [a, b] = inner.sample();
          (
            a.into_pyarray(py).into_py_any(py).unwrap(),
            b.into_pyarray(py).into_py_any(py).unwrap(),
          )
        } else if let Some(ref inner) = self.inner_f32 {
          let [a, b] = inner.sample();
          (
            a.into_pyarray(py).into_py_any(py).unwrap(),
            b.into_pyarray(py).into_py_any(py).unwrap(),
          )
        } else {
          unreachable!()
        }
      }

      fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
        use numpy::IntoPyArray;
        use numpy::ndarray::Array2;
        use crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        if let Some(ref inner) = self.inner_f64 {
          let samples = inner.sample_par(m);
          let n = samples[0][0].len();
          let mut r0 = Array2::<f64>::zeros((m, n));
          let mut r1 = Array2::<f64>::zeros((m, n));
          for (i, [a, b]) in samples.iter().enumerate() {
            r0.row_mut(i).assign(a);
            r1.row_mut(i).assign(b);
          }
          (
            r0.into_pyarray(py).into_py_any(py).unwrap(),
            r1.into_pyarray(py).into_py_any(py).unwrap(),
          )
        } else if let Some(ref inner) = self.inner_f32 {
          let samples = inner.sample_par(m);
          let n = samples[0][0].len();
          let mut r0 = Array2::<f32>::zeros((m, n));
          let mut r1 = Array2::<f32>::zeros((m, n));
          for (i, [a, b]) in samples.iter().enumerate() {
            r0.row_mut(i).assign(a);
            r1.row_mut(i).assign(b);
          }
          (
            r0.into_pyarray(py).into_py_any(py).unwrap(),
            r1.into_pyarray(py).into_py_any(py).unwrap(),
          )
        } else {
          unreachable!()
        }
      }
    }
  };
}

#[cfg(not(feature = "python"))]
macro_rules! py_process_2x1d {
  ($($tt:tt)*) => {};
}

#[cfg(feature = "python")]
macro_rules! py_process_2d {
  ($py_name:ident, $inner:ident,
    sig: ($($sig:tt)*),
    params: ($($param:ident : $pty:ty),* $(,)?)
  ) => {
    #[pyo3::prelude::pyclass]
    pub struct $py_name {
      inner_f32: Option<$inner<f32>>,
      inner_f64: Option<$inner<f64>>,
    }

    #[pyo3::prelude::pymethods]
    impl $py_name {
      #[new]
      #[pyo3(signature = ($($sig)*))]
      fn new($($param: $pty,)* dtype: Option<&str>) -> Self {
        match dtype.unwrap_or("f64") {
          "f32" => Self {
            inner_f32: Some($inner::new($(crate::python::IntoF32::into_f32($param)),*)),
            inner_f64: None,
          },
          _ => Self {
            inner_f32: None,
            inner_f64: Some($inner::new($(crate::python::IntoF64::into_f64($param)),*)),
          },
        }
      }

      fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
        use numpy::IntoPyArray;
        use crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        if let Some(ref inner) = self.inner_f64 {
          inner.sample().into_pyarray(py).into_py_any(py).unwrap()
        } else if let Some(ref inner) = self.inner_f32 {
          inner.sample().into_pyarray(py).into_py_any(py).unwrap()
        } else {
          unreachable!()
        }
      }

      fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
        use numpy::IntoPyArray;
        use crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        if let Some(ref inner) = self.inner_f64 {
          let samples = inner.sample_par(m);
          pyo3::types::PyList::new(
            py,
            samples.iter().map(|s| s.clone().into_pyarray(py).into_py_any(py).unwrap()),
          ).unwrap().into_py_any(py).unwrap()
        } else if let Some(ref inner) = self.inner_f32 {
          let samples = inner.sample_par(m);
          pyo3::types::PyList::new(
            py,
            samples.iter().map(|s| s.clone().into_pyarray(py).into_py_any(py).unwrap()),
          ).unwrap().into_py_any(py).unwrap()
        } else {
          unreachable!()
        }
      }
    }
  };
}

#[cfg(not(feature = "python"))]
macro_rules! py_process_2d {
  ($($tt:tt)*) => {};
}

#[cfg(feature = "python")]
macro_rules! py_distribution {
  ($py_name:ident, $inner:ident,
    sig: ($($sig:tt)*),
    params: ($($param:ident : $pty:ty),* $(,)?)
  ) => {
    #[pyo3::prelude::pyclass]
    pub struct $py_name {
      inner_f32: Option<std::sync::Mutex<$inner<f32>>>,
      inner_f64: Option<std::sync::Mutex<$inner<f64>>>,
    }

    #[pyo3::prelude::pymethods]
    impl $py_name {
      #[new]
      #[pyo3(signature = ($($sig)*))]
      fn new($($param: $pty,)* dtype: Option<&str>) -> Self {
        match dtype.unwrap_or("f64") {
          "f32" => Self {
            inner_f32: Some(std::sync::Mutex::new($inner::new($(crate::python::IntoF32::into_f32($param)),*))),
            inner_f64: None,
          },
          _ => Self {
            inner_f32: None,
            inner_f64: Some(std::sync::Mutex::new($inner::new($(crate::python::IntoF64::into_f64($param)),*))),
          },
        }
      }

      fn sample<'py>(&self, py: pyo3::Python<'py>, n: usize) -> pyo3::Py<pyo3::PyAny> {
        use rand_distr::Distribution;
        use numpy::IntoPyArray;
        use pyo3::IntoPyObjectExt;
        let mut rng = rand::rng();
        if let Some(ref mtx) = self.inner_f64 {
          let inner = mtx.lock().unwrap();
          let mut buf = vec![0.0f64; n];
          for v in buf.iter_mut() {
            *v = inner.sample(&mut rng);
          }
          numpy::ndarray::Array1::from_vec(buf).into_pyarray(py).into_py_any(py).unwrap()
        } else if let Some(ref mtx) = self.inner_f32 {
          let inner = mtx.lock().unwrap();
          let mut buf = vec![0.0f32; n];
          for v in buf.iter_mut() {
            *v = inner.sample(&mut rng);
          }
          numpy::ndarray::Array1::from_vec(buf).into_pyarray(py).into_py_any(py).unwrap()
        } else {
          unreachable!()
        }
      }
    }
  };
}

#[cfg(not(feature = "python"))]
macro_rules! py_distribution {
  ($($tt:tt)*) => {};
}

#[cfg(feature = "python")]
macro_rules! py_distribution_int {
  ($py_name:ident, $inner:ident,
    sig: ($($sig:tt)*),
    params: ($($param:ident : $pty:ty),* $(,)?)
  ) => {
    #[pyo3::prelude::pyclass]
    pub struct $py_name {
      inner: std::sync::Mutex<$inner<i64>>,
    }

    #[pyo3::prelude::pymethods]
    impl $py_name {
      #[new]
      #[pyo3(signature = ($($sig)*))]
      fn new($($param: $pty),*) -> Self {
        Self {
          inner: std::sync::Mutex::new($inner::new($($param),*)),
        }
      }

      fn sample<'py>(&self, py: pyo3::Python<'py>, n: usize) -> pyo3::Py<pyo3::PyAny> {
        use rand_distr::Distribution;
        use numpy::IntoPyArray;
        use pyo3::IntoPyObjectExt;
        let mut rng = rand::rng();
        let inner = self.inner.lock().unwrap();
        let mut buf = vec![0i64; n];
        for v in buf.iter_mut() {
          *v = inner.sample(&mut rng);
        }
        numpy::ndarray::Array1::from_vec(buf).into_pyarray(py).into_py_any(py).unwrap()
      }
    }
  };
}

#[cfg(not(feature = "python"))]
macro_rules! py_distribution_int {
  ($($tt:tt)*) => {};
}
