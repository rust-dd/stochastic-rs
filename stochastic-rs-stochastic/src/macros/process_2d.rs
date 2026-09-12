#[cfg(feature = "python")]
#[macro_export]
macro_rules! py_process_2d {
  ($py_name:ident, $inner:ident,
    sig: ($($sig:tt)*),
    params: ($($param:ident : $pty:ty),* $(,)?)
  ) => {
    #[pyo3::prelude::pyclass]
    pub struct $py_name {
      inner_f32: Option<$inner<f32>>,
      inner_f64: Option<$inner<f64>>,
      seeded_f32: Option<$inner<f32, stochastic_rs_core::simd_rng::Deterministic>>,
      seeded_f64: Option<$inner<f64, stochastic_rs_core::simd_rng::Deterministic>>,
    }

    #[pyo3::prelude::pymethods]
    impl $py_name {
      #[new]
      #[pyo3(signature = ($($sig)*))]
      fn new($($param: $pty,)* seed: Option<u64>, dtype: Option<&str>) -> Self {
        let mut s = Self { inner_f32: None, inner_f64: None, seeded_f32: None, seeded_f64: None };
        match (seed, dtype.unwrap_or("f64")) {
          (Some(sd), "f32") => { s.seeded_f32 = Some($inner::new($(stochastic_rs_core::python::IntoF32::into_f32($param),)* stochastic_rs_core::simd_rng::Deterministic::new(sd))); },
          (Some(sd), _) => { s.seeded_f64 = Some($inner::new($(stochastic_rs_core::python::IntoF64::into_f64($param),)* stochastic_rs_core::simd_rng::Deterministic::new(sd))); },
          (None, "f32") => { s.inner_f32 = Some($inner::new($(stochastic_rs_core::python::IntoF32::into_f32($param),)* stochastic_rs_core::simd_rng::Unseeded)); },
          (None, _) => { s.inner_f64 = Some($inner::new($(stochastic_rs_core::python::IntoF64::into_f64($param),)* stochastic_rs_core::simd_rng::Unseeded)); },
        }
        s
      }

      /// Why this configuration cannot run on a device kernel, or `None`
      /// when it can — a grid past a per-path array, a dimension past the
      /// state slots, a coefficient that is a closure, a mode with no grid.
      ///
      /// The question is about the configuration, not the handle, so the
      /// answer is the same whatever `device=` was passed, and a `device=`
      /// that cannot be honoured samples on the host at a cost this is the
      /// only way to see coming.
      fn device_fallback(&self) -> Option<&'static str> {
        use $crate::traits::ProcessExt;
        $crate::py_dispatch!(self, |inner| inner.device_fallback())
      }

      /// Whether this configuration runs on a device kernel: the absence of
      /// a [`device_fallback`](Self::device_fallback) reason.
      fn device_ready(&self) -> bool {
        self.device_fallback().is_none()
      }

      fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
        use numpy::IntoPyArray;
        use $crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        $crate::py_dispatch!(self, |inner| inner.sample().into_pyarray(py).into_py_any(py).unwrap())
      }

      /// Same reproducibility guarantee as `py_process_1d!`'s `sample_par`
      /// (see its doc comment) — bit-identical across rayon thread-pool
      /// sizes for a given seed and `m`.
      fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
        use numpy::IntoPyArray;
        use $crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        $crate::py_dispatch!(self, |inner| {
          let samples = inner.sample_par(m);
          pyo3::types::PyList::new(
            py,
            samples.iter().map(|s| s.clone().into_pyarray(py).into_py_any(py).unwrap()),
          ).unwrap().into_py_any(py).unwrap()
        })
      }
    }
  };

  ($py_name:ident, $inner:ident,
    sig: ($($sig:tt)*),
    params: ($($param:ident : $pty:ty),* $(,)?),
    device
  ) => {
    #[pyo3::prelude::pyclass]
    pub struct $py_name {
      inner_f32: Option<$inner<f32>>,
      inner_f64: Option<$inner<f64>>,
      seeded_f32: Option<$inner<f32, $crate::python_device::SharedSeed>>,
      seeded_f64: Option<$inner<f64, $crate::python_device::SharedSeed>>,
      device: $crate::python_device::Device,
    }

    #[pyo3::prelude::pymethods]
    impl $py_name {
      #[new]
      #[pyo3(signature = ($($sig)*, device=None))]
      fn new($($param: $pty,)* seed: Option<u64>, dtype: Option<&str>, device: Option<&str>) -> pyo3::PyResult<Self> {
        let device = $crate::python_device::Device::parse(device, dtype.unwrap_or("f64"))?;
        let mut s = Self { inner_f32: None, inner_f64: None, seeded_f32: None, seeded_f64: None, device };
        match (seed, dtype.unwrap_or("f64")) {
          (Some(sd), "f32") => { s.seeded_f32 = Some($inner::new($(stochastic_rs_core::python::IntoF32::into_f32($param),)* $crate::python_device::SharedSeed::new(sd))); },
          (Some(sd), _) => { s.seeded_f64 = Some($inner::new($(stochastic_rs_core::python::IntoF64::into_f64($param),)* $crate::python_device::SharedSeed::new(sd))); },
          (None, "f32") => { s.inner_f32 = Some($inner::new($(stochastic_rs_core::python::IntoF32::into_f32($param),)* stochastic_rs_core::simd_rng::Unseeded)); },
          (None, _) => { s.inner_f64 = Some($inner::new($(stochastic_rs_core::python::IntoF64::into_f64($param),)* stochastic_rs_core::simd_rng::Unseeded)); },
        }
        Ok(s)
      }

      /// Why this configuration cannot run on a device kernel, or `None`
      /// when it can — a grid past a per-path array, a dimension past the
      /// state slots, a coefficient that is a closure, a mode with no grid.
      ///
      /// The question is about the configuration, not the handle, so the
      /// answer is the same whatever `device=` was passed, and a `device=`
      /// that cannot be honoured samples on the host at a cost this is the
      /// only way to see coming.
      fn device_fallback(&self) -> Option<&'static str> {
        use $crate::traits::ProcessExt;
        $crate::py_dispatch!(self, |inner| inner.device_fallback())
      }

      /// Whether this configuration runs on a device kernel: the absence of
      /// a [`device_fallback`](Self::device_fallback) reason.
      fn device_ready(&self) -> bool {
        self.device_fallback().is_none()
      }

      fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
        use numpy::IntoPyArray;
        use $crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        $crate::py_device_dispatch!(self, |inner| inner.sample().into_pyarray(py).into_py_any(py).unwrap())
      }

      /// Same reproducibility guarantee as `py_process_1d!`'s `sample_par`
      /// (see its doc comment) — bit-identical across rayon thread-pool
      /// sizes for a given seed and `m`.
      fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
        use numpy::IntoPyArray;
        use $crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        $crate::py_device_dispatch!(self, |inner| {
          let samples = inner.sample_par(m);
          pyo3::types::PyList::new(
            py,
            samples.iter().map(|s| s.clone().into_pyarray(py).into_py_any(py).unwrap()),
          ).unwrap().into_py_any(py).unwrap()
        })
      }
    }
  };
}

#[cfg(not(feature = "python"))]
#[macro_export]
macro_rules! py_process_2d {
  ($($tt:tt)*) => {};
}
