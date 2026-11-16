use super::*;
use crate::python_device::Device;
use crate::python_device::SharedSeed;
use crate::python_params::drift_f32;
use crate::python_params::drift_from_kwargs;

#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyMjdLog {
  inner_f32: Option<MjdLog<f32>>,
  inner_f64: Option<MjdLog<f64>>,
  seeded_f32: Option<MjdLog<f32, SharedSeed>>,
  seeded_f64: Option<MjdLog<f64, SharedSeed>>,
  device: Device,
}

py_process_methods!(PyMjdLog, array1, py_device_dispatch, {
  #[new]
  #[pyo3(signature = (mu=None, b=None, r=None, r_f=None, *, sigma, lambda_, nu, omega, n, s0, t=None, seed=None, dtype=None, device=None))]
  fn new(
    mu: Option<f64>,
    b: Option<f64>,
    r: Option<f64>,
    r_f: Option<f64>,
    sigma: f64,
    lambda_: f64,
    nu: f64,
    omega: f64,
    n: usize,
    s0: f64,
    t: Option<f64>,
    seed: Option<u64>,
    dtype: Option<&str>,
    device: Option<&str>,
  ) -> pyo3::PyResult<Self> {
    let device = Device::parse(device, dtype.unwrap_or("f64"))?;
    stochastic_rs_distributions::python::value_error_on_panic(|| -> pyo3::PyResult<Self> {
      let drift = drift_from_kwargs(mu, b, r, r_f)?;
      let mut s = Self {
        inner_f32: None,
        inner_f64: None,
        seeded_f32: None,
        seeded_f64: None,
        device,
      };
      match (seed, dtype.unwrap_or("f64")) {
        (Some(sd), "f32") => {
          s.seeded_f32 = Some(MjdLog::new(
            drift_f32(drift),
            sigma as f32,
            lambda_ as f32,
            nu as f32,
            omega as f32,
            n,
            s0 as f32,
            t.map(|v| v as f32),
            SharedSeed::new(sd),
          ));
        }
        (Some(sd), _) => {
          s.seeded_f64 = Some(MjdLog::new(
            drift,
            sigma,
            lambda_,
            nu,
            omega,
            n,
            s0,
            t,
            SharedSeed::new(sd),
          ));
        }
        (None, "f32") => {
          s.inner_f32 = Some(MjdLog::new(
            drift_f32(drift),
            sigma as f32,
            lambda_ as f32,
            nu as f32,
            omega as f32,
            n,
            s0 as f32,
            t.map(|v| v as f32),
            Unseeded,
          ));
        }
        (None, _) => {
          s.inner_f64 = Some(MjdLog::new(
            drift, sigma, lambda_, nu, omega, n, s0, t, Unseeded,
          ));
        }
      }
      Ok(s)
    })?
  }
});
