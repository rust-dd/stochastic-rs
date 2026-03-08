//! # Macros
//!
//! $$
//! \text{macro expansion} : (\text{model spec}) \mapsto (\text{impl blocks})
//! $$
//!

/// Dispatch across 4-field Python wrappers (inner_f64, seeded_f64, inner_f32, seeded_f32).
#[cfg(feature = "python")]
macro_rules! py_dispatch {
  ($self:expr, |$inner:ident| $body:expr) => {
    if let Some(ref $inner) = $self.inner_f64 { $body }
    else if let Some(ref $inner) = $self.seeded_f64 { $body }
    else if let Some(ref $inner) = $self.inner_f32 { $body }
    else if let Some(ref $inner) = $self.seeded_f32 { $body }
    else { unreachable!() }
  };
}

/// Dispatch across 2-field f64-only Python wrappers (inner, seeded).
#[cfg(feature = "python")]
macro_rules! py_dispatch_f64 {
  ($self:expr, |$inner:ident| $body:expr) => {
    if let Some(ref $inner) = $self.inner { $body }
    else if let Some(ref $inner) = $self.seeded { $body }
    else { unreachable!() }
  };
}

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
      seeded_f32: Option<$inner<f32, crate::simd_rng::Deterministic>>,
      seeded_f64: Option<$inner<f64, crate::simd_rng::Deterministic>>,
    }

    #[pyo3::prelude::pymethods]
    impl $py_name {
      #[new]
      #[pyo3(signature = ($($sig)*))]
      fn new($($param: $pty,)* seed: Option<u64>, dtype: Option<&str>) -> Self {
        let mut s = Self { inner_f32: None, inner_f64: None, seeded_f32: None, seeded_f64: None };
        match (seed, dtype.unwrap_or("f64")) {
          (Some(sd), "f32") => { s.seeded_f32 = Some($inner::seeded($(crate::python::IntoF32::into_f32($param),)* sd)); },
          (Some(sd), _) => { s.seeded_f64 = Some($inner::seeded($(crate::python::IntoF64::into_f64($param),)* sd)); },
          (None, "f32") => { s.inner_f32 = Some($inner::new($(crate::python::IntoF32::into_f32($param)),*)); },
          (None, _) => { s.inner_f64 = Some($inner::new($(crate::python::IntoF64::into_f64($param)),*)); },
        }
        s
      }

      fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
        use numpy::IntoPyArray;
        use crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        py_dispatch!(self, |inner| inner.sample().into_pyarray(py).into_py_any(py).unwrap())
      }

      fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
        use numpy::IntoPyArray;
        use numpy::ndarray::Array2;
        use crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        py_dispatch!(self, |inner| {
          let paths = inner.sample_par(m);
          let n = paths[0].len();
          let mut result = Array2::zeros((m, n));
          for (i, path) in paths.iter().enumerate() {
            result.row_mut(i).assign(path);
          }
          result.into_pyarray(py).into_py_any(py).unwrap()
        })
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
      seeded_f32: Option<$inner<f32, crate::simd_rng::Deterministic>>,
      seeded_f64: Option<$inner<f64, crate::simd_rng::Deterministic>>,
    }

    #[pyo3::prelude::pymethods]
    impl $py_name {
      #[new]
      #[pyo3(signature = ($($sig)*))]
      fn new($($param: $pty,)* seed: Option<u64>, dtype: Option<&str>) -> Self {
        let mut s = Self { inner_f32: None, inner_f64: None, seeded_f32: None, seeded_f64: None };
        match (seed, dtype.unwrap_or("f64")) {
          (Some(sd), "f32") => { s.seeded_f32 = Some($inner::seeded($(crate::python::IntoF32::into_f32($param),)* sd)); },
          (Some(sd), _) => { s.seeded_f64 = Some($inner::seeded($(crate::python::IntoF64::into_f64($param),)* sd)); },
          (None, "f32") => { s.inner_f32 = Some($inner::new($(crate::python::IntoF32::into_f32($param)),*)); },
          (None, _) => { s.inner_f64 = Some($inner::new($(crate::python::IntoF64::into_f64($param)),*)); },
        }
        s
      }

      fn sample<'py>(&self, py: pyo3::Python<'py>) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
        use numpy::IntoPyArray;
        use crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        py_dispatch!(self, |inner| {
          let [a, b] = inner.sample();
          (a.into_pyarray(py).into_py_any(py).unwrap(), b.into_pyarray(py).into_py_any(py).unwrap())
        })
      }

      fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
        use numpy::IntoPyArray;
        use numpy::ndarray::Array2;
        use crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        py_dispatch!(self, |inner| {
          let samples = inner.sample_par(m);
          let n = samples[0][0].len();
          let mut r0 = Array2::zeros((m, n));
          let mut r1 = Array2::zeros((m, n));
          for (i, [a, b]) in samples.iter().enumerate() {
            r0.row_mut(i).assign(a);
            r1.row_mut(i).assign(b);
          }
          (r0.into_pyarray(py).into_py_any(py).unwrap(), r1.into_pyarray(py).into_py_any(py).unwrap())
        })
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
      seeded_f32: Option<$inner<f32, crate::simd_rng::Deterministic>>,
      seeded_f64: Option<$inner<f64, crate::simd_rng::Deterministic>>,
    }

    #[pyo3::prelude::pymethods]
    impl $py_name {
      #[new]
      #[pyo3(signature = ($($sig)*))]
      fn new($($param: $pty,)* seed: Option<u64>, dtype: Option<&str>) -> Self {
        let mut s = Self { inner_f32: None, inner_f64: None, seeded_f32: None, seeded_f64: None };
        match (seed, dtype.unwrap_or("f64")) {
          (Some(sd), "f32") => { s.seeded_f32 = Some($inner::seeded($(crate::python::IntoF32::into_f32($param),)* sd)); },
          (Some(sd), _) => { s.seeded_f64 = Some($inner::seeded($(crate::python::IntoF64::into_f64($param),)* sd)); },
          (None, "f32") => { s.inner_f32 = Some($inner::new($(crate::python::IntoF32::into_f32($param)),*)); },
          (None, _) => { s.inner_f64 = Some($inner::new($(crate::python::IntoF64::into_f64($param)),*)); },
        }
        s
      }

      fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
        use numpy::IntoPyArray;
        use crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        py_dispatch!(self, |inner| inner.sample().into_pyarray(py).into_py_any(py).unwrap())
      }

      fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
        use numpy::IntoPyArray;
        use crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        py_dispatch!(self, |inner| {
          let samples = inner.sample_par(m);
          pyo3::types::PyList::new(
            py,
            samples.iter().map(|s| s.clone().into_pyarray(py).into_py_any(py).unwrap()),
          ).unwrap().into_py_any(py).unwrap()
        })
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
    #[pyo3::prelude::pyclass(unsendable)]
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

      fn sample<'py>(&self, py: pyo3::Python<'py>, n: usize) -> pyo3::Py<pyo3::PyAny> {
        use crate::distributions::DistributionSampler;
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
        use crate::distributions::DistributionSampler;
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
macro_rules! py_distribution {
  ($($tt:tt)*) => {};
}

#[cfg(feature = "python")]
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
      fn new($($param: $pty),*) -> Self {
        Self {
          inner: $inner::new($($param),*),
        }
      }

      fn sample<'py>(&self, py: pyo3::Python<'py>, n: usize) -> pyo3::Py<pyo3::PyAny> {
        use crate::distributions::DistributionSampler;
        use numpy::IntoPyArray;
        use pyo3::IntoPyObjectExt;
        self.inner.sample_n(n).into_pyarray(py).into_py_any(py).unwrap()
      }

      fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize, n: usize) -> pyo3::Py<pyo3::PyAny> {
        use crate::distributions::DistributionSampler;
        use numpy::IntoPyArray;
        use pyo3::IntoPyObjectExt;
        self.inner.sample_matrix(m, n).into_pyarray(py).into_py_any(py).unwrap()
      }
    }
  };
}

#[cfg(not(feature = "python"))]
macro_rules! py_distribution_int {
  ($($tt:tt)*) => {};
}
