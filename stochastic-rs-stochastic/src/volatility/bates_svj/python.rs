use super::*;

#[pyo3::prelude::pyclass]
pub struct PyBatesSvj {
  inner_f32: Option<BatesSvj<f32>>,
  inner_f64: Option<BatesSvj<f64>>,
  seeded_f32: Option<BatesSvj<f32, crate::python_device::SharedSeed>>,
  seeded_f64: Option<BatesSvj<f64, crate::python_device::SharedSeed>>,
  /// The device the class samples on, chosen at construction.
  device: crate::python_device::Device,
}

#[pyo3::prelude::pymethods]
impl PyBatesSvj {
  #[new]
  #[pyo3(signature = (lambda_, nu, omega, alpha, beta, sigma, rho, n, mu=None, b=None, r=None, r_f=None, s0=None, v0=None, t=None, use_sym=None, seed=None, dtype=None, device=None))]
  fn new(
    lambda_: f64,
    nu: f64,
    omega: f64,
    alpha: f64,
    beta: f64,
    sigma: f64,
    rho: f64,
    n: usize,
    mu: Option<f64>,
    b: Option<f64>,
    r: Option<f64>,
    r_f: Option<f64>,
    s0: Option<f64>,
    v0: Option<f64>,
    t: Option<f64>,
    use_sym: Option<bool>,
    seed: Option<u64>,
    dtype: Option<&str>,
    device: Option<&str>,
  ) -> pyo3::PyResult<Self> {
    let device = crate::python_device::Device::parse(device, dtype.unwrap_or("f64"))?;
    let mut s = Self {
      inner_f32: None,
      inner_f64: None,
      seeded_f32: None,
      seeded_f64: None,
      device,
    };
    match (seed, dtype.unwrap_or("f64")) {
      (Some(sd), "f32") => {
        s.seeded_f32 = Some(BatesSvj::new(
          mu.map(|v| v as f32),
          b.map(|v| v as f32),
          r.map(|v| v as f32),
          r_f.map(|v| v as f32),
          lambda_ as f32,
          nu as f32,
          omega as f32,
          alpha as f32,
          beta as f32,
          sigma as f32,
          rho as f32,
          n,
          s0.map(|v| v as f32),
          v0.map(|v| v as f32),
          t.map(|v| v as f32),
          use_sym,
          crate::python_device::SharedSeed::new(sd),
        ));
      }
      (Some(sd), _) => {
        s.seeded_f64 = Some(BatesSvj::new(
          mu,
          b,
          r,
          r_f,
          lambda_,
          nu,
          omega,
          alpha,
          beta,
          sigma,
          rho,
          n,
          s0,
          v0,
          t,
          use_sym,
          crate::python_device::SharedSeed::new(sd),
        ));
      }
      (None, "f32") => {
        s.inner_f32 = Some(BatesSvj::new(
          mu.map(|v| v as f32),
          b.map(|v| v as f32),
          r.map(|v| v as f32),
          r_f.map(|v| v as f32),
          lambda_ as f32,
          nu as f32,
          omega as f32,
          alpha as f32,
          beta as f32,
          sigma as f32,
          rho as f32,
          n,
          s0.map(|v| v as f32),
          v0.map(|v| v as f32),
          t.map(|v| v as f32),
          use_sym,
          Unseeded,
        ));
      }
      (None, _) => {
        s.inner_f64 = Some(BatesSvj::new(
          mu, b, r, r_f, lambda_, nu, omega, alpha, beta, sigma, rho, n, s0, v0, t, use_sym,
          Unseeded,
        ));
      }
    }
    Ok(s)
  }

  /// The reason a device kernel cannot carry this configuration, if there is
  /// one: the engine states what it cannot do rather than doing it quietly on
  /// the host. The question is about the configuration, not the handle, so
  /// the answer is the same whatever `device=` was passed.
  fn device_fallback(&self) -> Option<&'static str> {
    use crate::traits::ProcessExt;
    crate::py_dispatch!(self, |inner| inner.device_fallback())
  }

  /// Whether this configuration runs on a device kernel: the absence of a
  /// [`device_fallback`](Self::device_fallback) reason.
  fn device_ready(&self) -> bool {
    self.device_fallback().is_none()
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_device_dispatch!(self, |inner| {
      let [a, b] = inner.sample();
      (
        a.into_pyarray(py).into_py_any(py).unwrap(),
        b.into_pyarray(py).into_py_any(py).unwrap(),
      )
    })
  }

  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
    use numpy::IntoPyArray;
    use numpy::ndarray::Array2;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_device_dispatch!(self, |inner| {
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
  }
}
