use super::*;
use crate::python_device::Device;
use crate::python_device::SharedSeed;
use crate::python_params::PyVarianceBoundary;
use crate::python_params::drift_f32;
use crate::python_params::drift_from_kwargs;

#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyHestonLog {
  inner_f32: Option<HestonLog<f32>>,
  inner_f64: Option<HestonLog<f64>>,
  seeded_f32: Option<HestonLog<f32, SharedSeed>>,
  seeded_f64: Option<HestonLog<f64, SharedSeed>>,
  device: Device,
}

py_process_methods!(PyHestonLog, pair, py_device_dispatch, {
  #[new]
  #[pyo3(signature = (mu=None, b=None, r=None, r_f=None, *, kappa, theta, xi, rho, n, s0=None, v0=None, t=None, boundary=None, seed=None, dtype=None, device=None))]
  fn new(
    mu: Option<f64>,
    b: Option<f64>,
    r: Option<f64>,
    r_f: Option<f64>,
    kappa: f64,
    theta: f64,
    xi: f64,
    rho: f64,
    n: usize,
    s0: Option<f64>,
    v0: Option<f64>,
    t: Option<f64>,
    boundary: Option<PyVarianceBoundary>,
    seed: Option<u64>,
    dtype: Option<&str>,
    device: Option<&str>,
  ) -> pyo3::PyResult<Self> {
    let device = Device::parse(device, dtype.unwrap_or("f64"))?;
    stochastic_rs_distributions::python::value_error_on_panic(|| -> pyo3::PyResult<Self> {
      let drift = drift_from_kwargs(mu, b, r, r_f)?;
      let boundary = crate::python_params::boundary(boundary);
      let mut s = Self {
        inner_f32: None,
        inner_f64: None,
        seeded_f32: None,
        seeded_f64: None,
        device,
      };
      match (seed, dtype.unwrap_or("f64")) {
        (Some(sd), "f32") => {
          s.seeded_f32 = Some(HestonLog::new(
            drift_f32(drift),
            kappa as f32,
            theta as f32,
            xi as f32,
            rho as f32,
            n,
            s0.map(|v| v as f32),
            v0.map(|v| v as f32),
            t.map(|v| v as f32),
            boundary,
            SharedSeed::new(sd),
          ));
        }
        (Some(sd), _) => {
          s.seeded_f64 = Some(HestonLog::new(
            drift,
            kappa,
            theta,
            xi,
            rho,
            n,
            s0,
            v0,
            t,
            boundary,
            SharedSeed::new(sd),
          ));
        }
        (None, "f32") => {
          s.inner_f32 = Some(HestonLog::new(
            drift_f32(drift),
            kappa as f32,
            theta as f32,
            xi as f32,
            rho as f32,
            n,
            s0.map(|v| v as f32),
            v0.map(|v| v as f32),
            t.map(|v| v as f32),
            boundary,
            Unseeded,
          ));
        }
        (None, _) => {
          s.inner_f64 = Some(HestonLog::new(
            drift, kappa, theta, xi, rho, n, s0, v0, t, boundary, Unseeded,
          ));
        }
      }
      Ok(s)
    })?
  }
});
