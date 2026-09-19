use super::*;

#[pyo3::prelude::pyclass]
pub struct PyBlackKarasinski {
  inner: Option<BlackKarasinski<f64>>,
  seeded: Option<BlackKarasinski<f64, crate::python_device::SharedSeed>>,
  /// The device the class samples on, chosen at construction.
  device: crate::python_device::Device,
}

#[pyo3::prelude::pymethods]
impl PyBlackKarasinski {
  #[new]
  #[pyo3(signature = (theta, a, sigma, n, r0=None, t=None, seed=None, device=None))]
  fn new(
    theta: pyo3::Py<pyo3::PyAny>,
    a: f64,
    sigma: f64,
    n: usize,
    r0: Option<f64>,
    t: Option<f64>,
    seed: Option<u64>,
    device: Option<&str>,
  ) -> pyo3::PyResult<Self> {
    let device = crate::python_device::Device::parse(device, "f64")?;
    Ok(match seed {
      Some(s) => Self {
        device,
        inner: None,
        seeded: Some(BlackKarasinski::new(
          Fn1D::Py(theta),
          a,
          sigma,
          n,
          r0,
          t,
          crate::python_device::SharedSeed::new(s),
        )),
      },
      None => Self {
        device,
        inner: Some(BlackKarasinski::new(
          Fn1D::Py(theta),
          a,
          sigma,
          n,
          r0,
          t,
          Unseeded,
        )),
        seeded: None,
      },
    })
  }

  /// The reason a device kernel cannot carry this configuration, if there is
  /// one: the engine states what it cannot do rather than doing it quietly on
  /// the host. The question is about the configuration, not the handle, so
  /// the answer is the same whatever `device=` was passed.
  fn device_fallback(&self) -> Option<&'static str> {
    use crate::traits::ProcessExt;
    crate::py_dispatch_f64!(self, |inner| inner.device_fallback())
  }

  /// Whether this configuration runs on a device kernel: the absence of a
  /// [`device_fallback`](Self::device_fallback) reason.
  fn device_ready(&self) -> bool {
    self.device_fallback().is_none()
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_device_dispatch_f64!(self, |inner| inner
      .sample()
      .into_pyarray(py)
      .into_py_any(py)
      .unwrap())
  }

  /// `m` independent paths stacked into an `(m, n)` array. The GIL is
  /// released while the paths are generated; every callable coefficient
  /// re-acquires it, so a Python function is called from rayon workers one
  /// at a time.
  fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
    use ndarray::Array2;
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_device_dispatch_f64!(self, |inner| {
      let paths = py.detach(|| inner.sample_par(m));
      let n = paths.first().map_or(0, |p| p.len());
      let mut result = Array2::zeros((m, n));
      for (i, path) in paths.iter().enumerate() {
        result.row_mut(i).assign(path);
      }
      result.into_pyarray(py).into_py_any(py).unwrap()
    })
  }
}
