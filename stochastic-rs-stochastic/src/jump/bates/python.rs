use super::*;

#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyBates {
  inner_f32: Option<Bates<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<Bates<f64, crate::traits::CallableDist<f64>>>,
  seeded_f32: Option<Bates<f32, crate::traits::CallableDist<f32>, crate::simd_rng::Deterministic>>,
  seeded_f64: Option<Bates<f64, crate::traits::CallableDist<f64>, crate::simd_rng::Deterministic>>,
}

#[pyo3::prelude::pymethods]
impl PyBates {
  #[new]
  #[pyo3(signature = (lambda_, k, alpha, beta, sigma, rho, distribution, n, mu=None, b=None, r=None, r_f=None, *, s0, v0, t=None, boundary=None, seed=None, dtype=None))]
  fn new(
    lambda_: f64,
    k: f64,
    alpha: f64,
    beta: f64,
    sigma: f64,
    rho: f64,
    distribution: pyo3::Py<pyo3::PyAny>,
    n: usize,
    mu: Option<f64>,
    b: Option<f64>,
    r: Option<f64>,
    r_f: Option<f64>,
    s0: f64,
    v0: f64,
    t: Option<f64>,
    boundary: Option<crate::python_params::PyVarianceBoundary>,
    seed: Option<u64>,
    dtype: Option<&str>,
  ) -> pyo3::PyResult<Self> {
    let boundary = crate::python_params::boundary(boundary);
    stochastic_rs_distributions::python::value_error_on_panic(|| -> pyo3::PyResult<Self> {
      let drift = crate::python_params::drift_from_kwargs(mu, b, r, r_f)?;
      let mut s = Self {
        inner_f32: None,
        inner_f64: None,
        seeded_f32: None,
        seeded_f64: None,
      };
      match dtype.unwrap_or("f64") {
        "f32" => {
          let drift = crate::python_params::drift_f32(drift);
          let jump_dist = crate::traits::CallableDist::new(distribution);
          match seed {
            Some(sd) => {
              s.seeded_f32 = Some(Bates::new(
                drift,
                lambda_ as f32,
                k as f32,
                alpha as f32,
                beta as f32,
                sigma as f32,
                rho as f32,
                jump_dist,
                n,
                s0 as f32,
                v0 as f32,
                t.map(|v| v as f32),
                boundary,
                Deterministic::new(sd),
              ));
            }
            None => {
              s.inner_f32 = Some(Bates::new(
                drift,
                lambda_ as f32,
                k as f32,
                alpha as f32,
                beta as f32,
                sigma as f32,
                rho as f32,
                jump_dist,
                n,
                s0 as f32,
                v0 as f32,
                t.map(|v| v as f32),
                boundary,
                Unseeded,
              ));
            }
          }
        }
        _ => {
          let jump_dist = crate::traits::CallableDist::new(distribution);
          match seed {
            Some(sd) => {
              s.seeded_f64 = Some(Bates::new(
                drift,
                lambda_,
                k,
                alpha,
                beta,
                sigma,
                rho,
                jump_dist,
                n,
                s0,
                v0,
                t,
                boundary,
                Deterministic::new(sd),
              ));
            }
            None => {
              s.inner_f64 = Some(Bates::new(
                drift, lambda_, k, alpha, beta, sigma, rho, jump_dist, n, s0, v0, t, boundary,
                Unseeded,
              ));
            }
          }
        }
      }
      Ok(s)
    })?
  }

  fn sample<'py>(
    &self,
    py: pyo3::Python<'py>,
  ) -> pyo3::PyResult<(pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>)> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_dispatch!(self, |inner| {
        let [a, b] = inner.sample();
        (
          a.into_pyarray(py).into_py_any(py).unwrap(),
          b.into_pyarray(py).into_py_any(py).unwrap(),
        )
      })
    })
  }

  /// `m` paths as a pair of `(m, n)` arrays, price and variance. The GIL is released while they are
  /// generated; the Python law runs on rayon workers, one call at a time.
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
      py_dispatch!(self, |inner| {
        let samples = py.detach(|| inner.sample_par(m));
        let n = samples.first().map_or(0, |p| p[0].len());
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

/// Bates (1996) with its lognormal jump `e^Z − 1`, `Z ~ N(nu, omega²)`, compensated by its mean.
#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyBatesSvj {
  inner_f32: Option<Bates<f32, LogNormalJump<f32>>>,
  inner_f64: Option<Bates<f64, LogNormalJump<f64>>>,
  seeded_f32: Option<Bates<f32, LogNormalJump<f32>, crate::python_device::SharedSeed>>,
  seeded_f64: Option<Bates<f64, LogNormalJump<f64>, crate::python_device::SharedSeed>>,
  device: crate::python_device::Device,
}

#[pyo3::prelude::pymethods]
impl PyBatesSvj {
  #[new]
  #[pyo3(signature = (lambda_, nu, omega, alpha, beta, sigma, rho, n, mu=None, b=None, r=None, r_f=None, *, s0, v0, t=None, boundary=None, seed=None, dtype=None, device=None))]
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
    s0: f64,
    v0: f64,
    t: Option<f64>,
    boundary: Option<crate::python_params::PyVarianceBoundary>,
    seed: Option<u64>,
    dtype: Option<&str>,
    device: Option<&str>,
  ) -> pyo3::PyResult<Self> {
    let boundary = crate::python_params::boundary(boundary);
    stochastic_rs_distributions::python::value_error_on_panic(|| -> pyo3::PyResult<Self> {
      let device = crate::python_device::Device::parse(device, dtype.unwrap_or("f64"))?;
      let drift = crate::python_params::drift_from_kwargs(mu, b, r, r_f)?;
      let mut s = Self {
        inner_f32: None,
        inner_f64: None,
        seeded_f32: None,
        seeded_f64: None,
        device,
      };
      match (seed, dtype.unwrap_or("f64")) {
        (Some(sd), "f32") => {
          let law = LogNormalJump::new(nu as f32, omega as f32);
          s.seeded_f32 = Some(Bates::new(
            crate::python_params::drift_f32(drift),
            lambda_ as f32,
            law.mean(),
            alpha as f32,
            beta as f32,
            sigma as f32,
            rho as f32,
            law,
            n,
            s0 as f32,
            v0 as f32,
            t.map(|v| v as f32),
            boundary,
            crate::python_device::SharedSeed::new(sd),
          ));
        }
        (Some(sd), _) => {
          let law = LogNormalJump::new(nu, omega);
          s.seeded_f64 = Some(Bates::new(
            drift,
            lambda_,
            law.mean(),
            alpha,
            beta,
            sigma,
            rho,
            law,
            n,
            s0,
            v0,
            t,
            boundary,
            crate::python_device::SharedSeed::new(sd),
          ));
        }
        (None, "f32") => {
          let law = LogNormalJump::new(nu as f32, omega as f32);
          s.inner_f32 = Some(Bates::new(
            crate::python_params::drift_f32(drift),
            lambda_ as f32,
            law.mean(),
            alpha as f32,
            beta as f32,
            sigma as f32,
            rho as f32,
            law,
            n,
            s0 as f32,
            v0 as f32,
            t.map(|v| v as f32),
            boundary,
            Unseeded,
          ));
        }
        (None, _) => {
          let law = LogNormalJump::new(nu, omega);
          s.inner_f64 = Some(Bates::new(
            drift,
            lambda_,
            law.mean(),
            alpha,
            beta,
            sigma,
            rho,
            law,
            n,
            s0,
            v0,
            t,
            boundary,
            Unseeded,
          ));
        }
      }
      Ok(s)
    })?
  }

  /// Why this configuration cannot run on a device kernel, or `None` when it can; the
  /// configuration decides, so `device=` does not change the answer.
  fn device_fallback(&self) -> Option<&'static str> {
    use crate::traits::ProcessExt;
    crate::py_dispatch!(self, |inner| inner.device_fallback())
  }

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
      py_device_dispatch!(self, |inner| {
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
      py_device_dispatch!(self, |inner| {
        let samples = inner.sample_par(m);
        let n = samples.first().map_or(0, |p| p[0].len());
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
