use super::*;
use crate::python_device::Device;
use crate::python_device::SharedSeed;
use crate::python_params::horizon_f32;
use crate::python_params::horizon_from_kwargs;

#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyPoisson {
  inner_f32: Option<Poisson<f32>>,
  inner_f64: Option<Poisson<f64>>,
  seeded_f32: Option<Poisson<f32, SharedSeed>>,
  seeded_f64: Option<Poisson<f64, SharedSeed>>,
  device: Device,
}

py_process_methods!(PyPoisson, array1, py_device_dispatch, {
  #[new]
  #[pyo3(signature = (lambda_, n=None, t_max=None, seed=None, dtype=None, device=None))]
  fn new(
    lambda_: f64,
    n: Option<usize>,
    t_max: Option<f64>,
    seed: Option<u64>,
    dtype: Option<&str>,
    device: Option<&str>,
  ) -> pyo3::PyResult<Self> {
    let device = Device::parse(device, dtype.unwrap_or("f64"))?;
    stochastic_rs_distributions::python::value_error_on_panic(|| -> pyo3::PyResult<Self> {
      let horizon = horizon_from_kwargs(n, t_max)?;
      let mut s = Self {
        inner_f32: None,
        inner_f64: None,
        seeded_f32: None,
        seeded_f64: None,
        device,
      };
      match (seed, dtype.unwrap_or("f64")) {
        (Some(sd), "f32") => {
          s.seeded_f32 = Some(Poisson::new(
            lambda_ as f32,
            horizon_f32(horizon),
            SharedSeed::new(sd),
          ));
        }
        (Some(sd), _) => s.seeded_f64 = Some(Poisson::new(lambda_, horizon, SharedSeed::new(sd))),
        (None, "f32") => {
          s.inner_f32 = Some(Poisson::new(lambda_ as f32, horizon_f32(horizon), Unseeded));
        }
        (None, _) => s.inner_f64 = Some(Poisson::new(lambda_, horizon, Unseeded)),
      }
      Ok(s)
    })?
  }
});
