/// A process class's `#[pymethods]` block: the `#[new]` it is handed, then `device_fallback`,
/// `device_ready`, `sample` and `sample_par` for an `array1`, `array2` or `pair` output.
#[cfg(feature = "python")]
macro_rules! py_process_methods {
  ($py_name:ident, array1, $dispatch:ident, { $($new:tt)* }) => {
    py_process_methods!(@block $py_name, array1, $dispatch, (pyo3::Py<pyo3::PyAny>), { $($new)* });
  };

  ($py_name:ident, array2, $dispatch:ident, { $($new:tt)* }) => {
    py_process_methods!(@block $py_name, array2, $dispatch, (pyo3::Py<pyo3::PyAny>), { $($new)* });
  };

  ($py_name:ident, pair, $dispatch:ident, { $($new:tt)* }) => {
    py_process_methods!(
      @block $py_name, pair, $dispatch, ((pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>)), { $($new)* }
    );
  };

  (@block $py_name:ident, $shape:ident, $dispatch:ident, ($($out:tt)*), { $($new:tt)* }) => {
    #[pyo3::prelude::pymethods]
    impl $py_name {
      $($new)*

      /// Why this configuration cannot run on a device kernel, or `None` when it can; the
      /// configuration decides, so `device=` does not change the answer.
      fn device_fallback(&self) -> Option<&'static str> {
        use $crate::traits::ProcessExt;
        py_dispatch!(self, |inner| inner.device_fallback())
      }

      /// Whether this configuration runs on a device kernel: no
      /// [`device_fallback`](Self::device_fallback) reason.
      fn device_ready(&self) -> bool {
        self.device_fallback().is_none()
      }

      fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<$($out)*> {
        stochastic_rs_distributions::python::runtime_error_on_panic(|| {
          use numpy::IntoPyArray;
          use pyo3::IntoPyObjectExt;
          use $crate::traits::ProcessExt;
          $dispatch!(self, |inner| py_process_methods!(@one $shape, py, inner.sample()))
        })
      }

      /// `m` independent paths, bit-identical across rayon thread-pool sizes for a seed and `m`.
      fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::PyResult<$($out)*> {
        stochastic_rs_distributions::python::runtime_error_on_panic(|| {
          use numpy::IntoPyArray;
          use pyo3::IntoPyObjectExt;
          use $crate::traits::ProcessExt;
          $dispatch!(self, |inner| {
            py_process_methods!(@many $shape, py, m, inner.sample_par(m))
          })
        })
      }
    }
  };

  (@one array1, $py:ident, $path:expr) => {
    $path.into_pyarray($py).into_py_any($py).unwrap()
  };

  (@one array2, $py:ident, $path:expr) => {
    $path.into_pyarray($py).into_py_any($py).unwrap()
  };

  (@one pair, $py:ident, $path:expr) => {{
    let [a, b] = $path;
    (a.into_pyarray($py).into_py_any($py).unwrap(), b.into_pyarray($py).into_py_any($py).unwrap())
  }};

  (@many array1, $py:ident, $m:ident, $paths:expr) => {{
    let paths = $paths;
    let n = paths[0].len();
    let mut result = numpy::ndarray::Array2::zeros(($m, n));
    for (i, path) in paths.iter().enumerate() {
      result.row_mut(i).assign(path);
    }
    result.into_pyarray($py).into_py_any($py).unwrap()
  }};

  (@many array2, $py:ident, $m:ident, $paths:expr) => {{
    let samples = $paths;
    pyo3::types::PyList::new(
      $py,
      samples.iter().map(|s| s.clone().into_pyarray($py).into_py_any($py).unwrap()),
    ).unwrap().into_py_any($py).unwrap()
  }};

  (@many pair, $py:ident, $m:ident, $paths:expr) => {{
    let samples = $paths;
    let n = samples[0][0].len();
    let mut r0 = numpy::ndarray::Array2::zeros(($m, n));
    let mut r1 = numpy::ndarray::Array2::zeros(($m, n));
    for (i, [a, b]) in samples.iter().enumerate() {
      r0.row_mut(i).assign(a);
      r1.row_mut(i).assign(b);
    }
    (r0.into_pyarray($py).into_py_any($py).unwrap(), r1.into_pyarray($py).into_py_any($py).unwrap())
  }};
}
