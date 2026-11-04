use super::*;

/// `scheme` is `"euler"` (default) or `"qe"`, Andersen's (2008) quadratic-exponential variance step.
#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyHeston {
  inner_f32: Option<Heston<f32>>,
  inner_f64: Option<Heston<f64>>,
  seeded_f32: Option<Heston<f32, crate::python_device::SharedSeed>>,
  seeded_f64: Option<Heston<f64, crate::python_device::SharedSeed>>,
  /// The device the class samples on, chosen at construction.
  device: crate::python_device::Device,
  qe: Option<PyHestonQe>,
}

/// The [`AndersenQe`] twins of [`PyHeston`]'s slots, named for the dispatch macros.
struct PyHestonQe {
  inner_f32: Option<Heston<f32, Unseeded, AndersenQe>>,
  inner_f64: Option<Heston<f64, Unseeded, AndersenQe>>,
  seeded_f32: Option<Heston<f32, crate::python_device::SharedSeed, AndersenQe>>,
  seeded_f64: Option<Heston<f64, crate::python_device::SharedSeed, AndersenQe>>,
  device: crate::python_device::Device,
}

macro_rules! py_heston_dispatch {
  ($dispatch:ident, $self:expr, |$p:ident| $body:expr) => {
    match &$self.qe {
      Some(qe) => $crate::$dispatch!(qe, |$p| $body),
      None => $crate::$dispatch!($self, |$p| $body),
    }
  };
}

#[pyo3::prelude::pymethods]
impl PyHeston {
  #[new]
  #[pyo3(signature = (kappa, theta, sigma, rho, mu, n, s0=None, v0=None, t=None, pow=None, use_sym=None, seed=None, dtype=None, device=None, scheme="euler"))]
  fn new(
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    mu: f64,
    n: usize,
    s0: Option<f64>,
    v0: Option<f64>,
    t: Option<f64>,
    pow: Option<&str>,
    use_sym: Option<bool>,
    seed: Option<u64>,
    dtype: Option<&str>,
    device: Option<&str>,
    scheme: &str,
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> pyo3::PyResult<Self> {
      let device = crate::python_device::Device::parse(device, dtype.unwrap_or("f64"))?;
      let hp = match pow.unwrap_or("sqrt") {
        "three_halves" | "3/2" => HestonPow::ThreeHalves,
        _ => HestonPow::Sqrt,
      };
      let qe = match scheme.to_ascii_lowercase().as_str() {
        "euler" => false,
        "qe" => true,
        other => {
          return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "scheme must be 'euler' or 'qe', got '{other}'"
          )));
        }
      };
      let mut s = Self {
        inner_f32: None,
        inner_f64: None,
        seeded_f32: None,
        seeded_f64: None,
        device,
        qe: None,
      };
      match (seed, dtype.unwrap_or("f64")) {
        (Some(sd), "f32") => {
          s.seeded_f32 = Some(Heston::new(
            s0.map(|v| v as f32),
            v0.map(|v| v as f32),
            kappa as f32,
            theta as f32,
            sigma as f32,
            rho as f32,
            mu as f32,
            n,
            t.map(|v| v as f32),
            hp,
            use_sym,
            crate::python_device::SharedSeed::new(sd),
          ));
        }
        (Some(sd), _) => {
          s.seeded_f64 = Some(Heston::new(
            s0,
            v0,
            kappa,
            theta,
            sigma,
            rho,
            mu,
            n,
            t,
            hp,
            use_sym,
            crate::python_device::SharedSeed::new(sd),
          ));
        }
        (None, "f32") => {
          s.inner_f32 = Some(Heston::new(
            s0.map(|v| v as f32),
            v0.map(|v| v as f32),
            kappa as f32,
            theta as f32,
            sigma as f32,
            rho as f32,
            mu as f32,
            n,
            t.map(|v| v as f32),
            hp,
            use_sym,
            Unseeded,
          ));
        }
        (None, _) => {
          s.inner_f64 = Some(Heston::new(
            s0, v0, kappa, theta, sigma, rho, mu, n, t, hp, use_sym, Unseeded,
          ));
        }
      }
      if qe {
        s.qe = Some(PyHestonQe {
          inner_f32: s.inner_f32.take().map(Heston::qe),
          inner_f64: s.inner_f64.take().map(Heston::qe),
          seeded_f32: s.seeded_f32.take().map(Heston::qe),
          seeded_f64: s.seeded_f64.take().map(Heston::qe),
          device: s.device,
        });
      }
      Ok(s)
    })?
  }

  /// The reason a device kernel cannot carry this configuration, if there is
  /// one: the engine states what it cannot do rather than doing it quietly on
  /// the host. The question is about the configuration, not the handle, so
  /// the answer is the same whatever `device=` was passed.
  fn device_fallback(&self) -> Option<&'static str> {
    use crate::traits::ProcessExt;
    py_heston_dispatch!(py_dispatch, self, |inner| inner.device_fallback())
  }

  /// Whether this configuration runs on a device kernel: the absence of a
  /// [`device_fallback`](Self::device_fallback) reason.
  fn device_ready(&self) -> bool {
    self.device_fallback().is_none()
  }

  fn sample<'py>(
    &self,
    py: pyo3::Python<'py>,
  ) -> pyo3::PyResult<(pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>)> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_heston_dispatch!(py_device_dispatch, self, |inner| {
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
      py_heston_dispatch!(py_device_dispatch, self, |inner| {
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
