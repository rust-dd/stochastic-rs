#[cfg(feature = "python")]
macro_rules! py_process_1d {
  ($py_name:ident, $inner:ident,
    sig: ($($sig:tt)*),
    params: ($($param:ident : $pty:ty $(=> $map:path)?),* $(,)?)
  ) => {
    #[doc(hidden)]
    #[pyo3::prelude::pyclass]
    pub struct $py_name {
      inner_f32: Option<$inner<f32>>,
      inner_f64: Option<$inner<f64>>,
      seeded_f32: Option<$inner<f32, stochastic_rs_core::simd_rng::Deterministic>>,
      seeded_f64: Option<$inner<f64, stochastic_rs_core::simd_rng::Deterministic>>,
    }

    py_process_methods!($py_name, array1, py_dispatch, {
      #[new]
      #[pyo3(signature = ($($sig)*))]
      fn new($($param: $pty,)* seed: Option<u64>, dtype: Option<&str>) -> pyo3::PyResult<Self> {
        stochastic_rs_distributions::python::value_error_on_panic(|| {
          let mut s = Self { inner_f32: None, inner_f64: None, seeded_f32: None, seeded_f64: None };
          match (seed, dtype.unwrap_or("f64")) {
            (Some(sd), "f32") => { s.seeded_f32 = Some($inner::new($(py_arg!(f32, $param $(, $map)?),)* stochastic_rs_core::simd_rng::Deterministic::new(sd))); },
            (Some(sd), _) => { s.seeded_f64 = Some($inner::new($(py_arg!(f64, $param $(, $map)?),)* stochastic_rs_core::simd_rng::Deterministic::new(sd))); },
            (None, "f32") => { s.inner_f32 = Some($inner::new($(py_arg!(f32, $param $(, $map)?),)* stochastic_rs_core::simd_rng::Unseeded)); },
            (None, _) => { s.inner_f64 = Some($inner::new($(py_arg!(f64, $param $(, $map)?),)* stochastic_rs_core::simd_rng::Unseeded)); },
          }
          s
        })
      }
    });
  };

  ($py_name:ident, $inner:ident,
    sig: ($($sig:tt)*),
    params: ($($param:ident : $pty:ty $(=> $map:path)?),* $(,)?),
    device
  ) => {
    #[doc(hidden)]
    #[pyo3::prelude::pyclass]
    pub struct $py_name {
      inner_f32: Option<$inner<f32>>,
      inner_f64: Option<$inner<f64>>,
      seeded_f32: Option<$inner<f32, $crate::python_device::SharedSeed>>,
      seeded_f64: Option<$inner<f64, $crate::python_device::SharedSeed>>,
      device: $crate::python_device::Device,
    }

    py_process_methods!($py_name, array1, py_device_dispatch, {
      #[new]
      #[pyo3(signature = ($($sig)*, device=None))]
      fn new($($param: $pty,)* seed: Option<u64>, dtype: Option<&str>, device: Option<&str>) -> pyo3::PyResult<Self> {
        let device = $crate::python_device::Device::parse(device, dtype.unwrap_or("f64"))?;
        stochastic_rs_distributions::python::value_error_on_panic(|| {
          let mut s = Self { inner_f32: None, inner_f64: None, seeded_f32: None, seeded_f64: None, device };
          match (seed, dtype.unwrap_or("f64")) {
            (Some(sd), "f32") => { s.seeded_f32 = Some($inner::new($(py_arg!(f32, $param $(, $map)?),)* $crate::python_device::SharedSeed::new(sd))); },
            (Some(sd), _) => { s.seeded_f64 = Some($inner::new($(py_arg!(f64, $param $(, $map)?),)* $crate::python_device::SharedSeed::new(sd))); },
            (None, "f32") => { s.inner_f32 = Some($inner::new($(py_arg!(f32, $param $(, $map)?),)* stochastic_rs_core::simd_rng::Unseeded)); },
            (None, _) => { s.inner_f64 = Some($inner::new($(py_arg!(f64, $param $(, $map)?),)* stochastic_rs_core::simd_rng::Unseeded)); },
          }
          s
        })
      }
    });
  };
}

#[cfg(not(feature = "python"))]
macro_rules! py_process_1d {
  ($($tt:tt)*) => {};
}
