use super::*;
use crate::python_device::Device;
use crate::python_device::SharedSeed;
use crate::python_params::horizon_f32;
use crate::python_params::horizon_from_kwargs;

#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyHawkes {
  inner_f32: Option<Hawkes<f32>>,
  inner_f64: Option<Hawkes<f64>>,
  seeded_f32: Option<Hawkes<f32, SharedSeed>>,
  seeded_f64: Option<Hawkes<f64, SharedSeed>>,
  device: Device,
}

py_process_methods!(PyHawkes, array1, py_device_dispatch, {
  #[new]
  #[pyo3(signature = (mu, alpha, beta, n=None, t_max=None, seed=None, dtype=None, device=None))]
  fn new(
    mu: f64,
    alpha: f64,
    beta: f64,
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
      let (mu32, alpha32, beta32) = (mu as f32, alpha as f32, beta as f32);
      match (seed, dtype.unwrap_or("f64")) {
        (Some(sd), "f32") => {
          let seed = SharedSeed::new(sd);
          s.seeded_f32 = Some(Hawkes::new(
            mu32,
            alpha32,
            beta32,
            horizon_f32(horizon),
            seed,
          ));
        }
        (Some(sd), _) => {
          s.seeded_f64 = Some(Hawkes::new(mu, alpha, beta, horizon, SharedSeed::new(sd)));
        }
        (None, "f32") => {
          s.inner_f32 = Some(Hawkes::new(
            mu32,
            alpha32,
            beta32,
            horizon_f32(horizon),
            Unseeded,
          ));
        }
        (None, _) => s.inner_f64 = Some(Hawkes::new(mu, alpha, beta, horizon, Unseeded)),
      }
      Ok(s)
    })?
  }
});
