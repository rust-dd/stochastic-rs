//! PyO3 wrapper macros for stochastic process types.

#[cfg(feature = "python")]
#[macro_export]
macro_rules! py_dispatch_f64 {
  ($self:expr, |$inner:ident| $body:expr) => {
    if let Some(ref $inner) = $self.inner {
      $body
    } else if let Some(ref $inner) = $self.seeded {
      $body
    } else {
      unreachable!()
    }
  };
}

#[cfg(feature = "python")]
#[macro_export]
macro_rules! py_dispatch {
  ($self:expr, |$inner:ident| $body:expr) => {
    if let Some(ref $inner) = $self.inner_f64 {
      $body
    } else if let Some(ref $inner) = $self.seeded_f64 {
      $body
    } else if let Some(ref $inner) = $self.inner_f32 {
      $body
    } else if let Some(ref $inner) = $self.seeded_f32 {
      $body
    } else {
      unreachable!()
    }
  };
}

/// One `sample*` body over a `float64` slot on the class's device: the host
/// slot as is, a device slot re-typed with `.on::<B>()` on a clone. The
/// single-precision devices never reach here — `Device::parse` rejects them
/// for a `float64` process.
#[cfg(feature = "python")]
#[macro_export]
macro_rules! py_on_device_f64 {
  ($device:expr, $inner:ident, |$p:ident| $body:expr) => {
    match $device {
      $crate::python_device::Device::Cpu => {
        let $p = $inner;
        $body
      }
      #[cfg(feature = "accelerate")]
      $crate::python_device::Device::Accelerate => {
        let owned = $inner.clone().on_device($crate::device::Accelerate);
        let $p = &owned;
        $body
      }
      #[cfg(feature = "cuda-native")]
      $crate::python_device::Device::CudaNative(ordinal) => {
        let owned = $inner
          .clone()
          .on_device($crate::device::CudaNative::new(ordinal));
        let $p = &owned;
        $body
      }
      #[allow(unreachable_patterns)]
      _ => unreachable!("single-precision devices are rejected for float64 at construction"),
    }
  };
}

/// [`py_on_device_f64!`] for a `float32` slot, which every compiled device
/// accepts.
#[cfg(feature = "python")]
#[macro_export]
macro_rules! py_on_device_f32 {
  ($device:expr, $inner:ident, |$p:ident| $body:expr) => {
    match $device {
      $crate::python_device::Device::Cpu => {
        let $p = $inner;
        $body
      }
      #[cfg(feature = "accelerate")]
      $crate::python_device::Device::Accelerate => {
        let owned = $inner.clone().on_device($crate::device::Accelerate);
        let $p = &owned;
        $body
      }
      #[cfg(feature = "cuda-native")]
      $crate::python_device::Device::CudaNative(ordinal) => {
        let owned = $inner
          .clone()
          .on_device($crate::device::CudaNative::new(ordinal));
        let $p = &owned;
        $body
      }
      #[cfg(feature = "metal")]
      $crate::python_device::Device::MetalNative(ordinal) => {
        let owned = $inner
          .clone()
          .on_device($crate::device::MetalNative::new(ordinal));
        let $p = &owned;
        $body
      }
      #[cfg(any(feature = "cubecl-cuda", feature = "cubecl-wgpu"))]
      $crate::python_device::Device::CubeCl(ordinal) => {
        let owned = $inner
          .clone()
          .on_device($crate::device::CubeCl::new(ordinal));
        let $p = &owned;
        $body
      }
      #[allow(unreachable_patterns)]
      _ => unreachable!("devices this build lacks are rejected at construction"),
    }
  };
}

/// [`py_dispatch!`] for a class with a `device` field: the four slots, each
/// routed through [`py_on_device_f64!`] / [`py_on_device_f32!`].
#[cfg(feature = "python")]
#[macro_export]
macro_rules! py_device_dispatch {
  ($self:expr, |$p:ident| $body:expr) => {
    if let Some(ref inner) = $self.inner_f64 {
      $crate::py_on_device_f64!($self.device, inner, |$p| $body)
    } else if let Some(ref inner) = $self.seeded_f64 {
      $crate::py_on_device_f64!($self.device, inner, |$p| $body)
    } else if let Some(ref inner) = $self.inner_f32 {
      $crate::py_on_device_f32!($self.device, inner, |$p| $body)
    } else if let Some(ref inner) = $self.seeded_f32 {
      $crate::py_on_device_f32!($self.device, inner, |$p| $body)
    } else {
      unreachable!()
    }
  };
}

#[cfg(feature = "python")]
#[macro_export]
macro_rules! py_process_1d {
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

      fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
        use numpy::IntoPyArray;
        use $crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        $crate::py_dispatch!(self, |inner| inner.sample().into_pyarray(py).into_py_any(py).unwrap())
      }

      /// `m` independent paths via [`ProcessExt::sample_par`], stacked into
      /// an `(m, n)` array. Bit-identical across rayon thread-pool sizes for
      /// a given seed and `m` — see `ProcessExt::sample_par`'s own doc and
      /// the 124-type guard in `tests/reproducibility_all_processes.rs` for
      /// the Rust-side proof, and
      /// `stochastic-rs-py/tests/test_sample_par_thread_count.py` for the
      /// same property verified across this PyO3 boundary (subprocesses
      /// with different `RAYON_NUM_THREADS` values). Until that Python test
      /// existed, the seeded path here serialized into `m` sequential
      /// `sample()` calls instead of calling `sample_par` — a different,
      /// also-deterministic sequence.
      fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
        use numpy::IntoPyArray;
        use numpy::ndarray::Array2;
        use $crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        $crate::py_dispatch!(self, |inner| {
          let paths = inner.sample_par(m);
          let n = paths[0].len();
          let mut result = Array2::zeros((m, n));
          for (i, path) in paths.iter().enumerate() {
            result.row_mut(i).assign(path);
          }
          result.into_pyarray(py).into_py_any(py).unwrap()
        })
      }
    }
  };

  // Same class with a `device=` argument (see `python_device::Device`).
  ($py_name:ident, $inner:ident,
    sig: ($($sig:tt)*),
    params: ($($param:ident : $pty:ty),* $(,)?),
    device
  ) => {
    #[pyo3::prelude::pyclass]
    pub struct $py_name {
      inner_f32: Option<$inner<f32>>,
      inner_f64: Option<$inner<f64>>,
      seeded_f32: Option<$inner<f32, stochastic_rs_core::simd_rng::Deterministic>>,
      seeded_f64: Option<$inner<f64, stochastic_rs_core::simd_rng::Deterministic>>,
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
          (Some(sd), "f32") => { s.seeded_f32 = Some($inner::new($(stochastic_rs_core::python::IntoF32::into_f32($param),)* stochastic_rs_core::simd_rng::Deterministic::new(sd))); },
          (Some(sd), _) => { s.seeded_f64 = Some($inner::new($(stochastic_rs_core::python::IntoF64::into_f64($param),)* stochastic_rs_core::simd_rng::Deterministic::new(sd))); },
          (None, "f32") => { s.inner_f32 = Some($inner::new($(stochastic_rs_core::python::IntoF32::into_f32($param),)* stochastic_rs_core::simd_rng::Unseeded)); },
          (None, _) => { s.inner_f64 = Some($inner::new($(stochastic_rs_core::python::IntoF64::into_f64($param),)* stochastic_rs_core::simd_rng::Unseeded)); },
        }
        Ok(s)
      }

      fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
        use numpy::IntoPyArray;
        use $crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        $crate::py_device_dispatch!(self, |inner| inner.sample().into_pyarray(py).into_py_any(py).unwrap())
      }

      /// `m` independent paths via [`ProcessExt::sample_par`], stacked into
      /// an `(m, n)` array. Bit-identical across rayon thread-pool sizes for
      /// a given seed and `m` — see `ProcessExt::sample_par`'s own doc and
      /// the 124-type guard in `tests/reproducibility_all_processes.rs` for
      /// the Rust-side proof, and
      /// `stochastic-rs-py/tests/test_sample_par_thread_count.py` for the
      /// same property verified across this PyO3 boundary (subprocesses
      /// with different `RAYON_NUM_THREADS` values). Until that Python test
      /// existed, the seeded path here serialized into `m` sequential
      /// `sample()` calls instead of calling `sample_par` — a different,
      /// also-deterministic sequence.
      fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
        use numpy::IntoPyArray;
        use numpy::ndarray::Array2;
        use $crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        $crate::py_device_dispatch!(self, |inner| {
          let paths = inner.sample_par(m);
          let n = paths[0].len();
          let mut result = Array2::zeros((m, n));
          for (i, path) in paths.iter().enumerate() {
            result.row_mut(i).assign(path);
          }
          result.into_pyarray(py).into_py_any(py).unwrap()
        })
      }
    }
  };
}

#[cfg(not(feature = "python"))]
#[macro_export]
macro_rules! py_process_1d {
  ($($tt:tt)*) => {};
}

#[cfg(feature = "python")]
#[macro_export]
macro_rules! py_process_2x1d {
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

      fn sample<'py>(&self, py: pyo3::Python<'py>) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
        use numpy::IntoPyArray;
        use $crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        $crate::py_dispatch!(self, |inner| {
          let [a, b] = inner.sample();
          (a.into_pyarray(py).into_py_any(py).unwrap(), b.into_pyarray(py).into_py_any(py).unwrap())
        })
      }

      /// Same reproducibility guarantee as `py_process_1d!`'s `sample_par`
      /// (see its doc comment) — bit-identical across rayon thread-pool
      /// sizes for a given seed and `m`.
      fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
        use numpy::IntoPyArray;
        use numpy::ndarray::Array2;
        use $crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        $crate::py_dispatch!(self, |inner| {
          let samples = inner.sample_par(m);
          let n = samples[0][0].len();
          let mut r0 = Array2::zeros((m, n));
          let mut r1 = Array2::zeros((m, n));
          for (i, [a, b]) in samples.iter().enumerate() {
            r0.row_mut(i).assign(a);
            r1.row_mut(i).assign(b);
          }
          (r0.into_pyarray(py).into_py_any(py).unwrap(), r1.into_pyarray(py).into_py_any(py).unwrap())
        })
      }
    }
  };

  // Same class with a `device=` argument (see `python_device::Device`).
  ($py_name:ident, $inner:ident,
    sig: ($($sig:tt)*),
    params: ($($param:ident : $pty:ty),* $(,)?),
    device
  ) => {
    #[pyo3::prelude::pyclass]
    pub struct $py_name {
      inner_f32: Option<$inner<f32>>,
      inner_f64: Option<$inner<f64>>,
      seeded_f32: Option<$inner<f32, stochastic_rs_core::simd_rng::Deterministic>>,
      seeded_f64: Option<$inner<f64, stochastic_rs_core::simd_rng::Deterministic>>,
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
          (Some(sd), "f32") => { s.seeded_f32 = Some($inner::new($(stochastic_rs_core::python::IntoF32::into_f32($param),)* stochastic_rs_core::simd_rng::Deterministic::new(sd))); },
          (Some(sd), _) => { s.seeded_f64 = Some($inner::new($(stochastic_rs_core::python::IntoF64::into_f64($param),)* stochastic_rs_core::simd_rng::Deterministic::new(sd))); },
          (None, "f32") => { s.inner_f32 = Some($inner::new($(stochastic_rs_core::python::IntoF32::into_f32($param),)* stochastic_rs_core::simd_rng::Unseeded)); },
          (None, _) => { s.inner_f64 = Some($inner::new($(stochastic_rs_core::python::IntoF64::into_f64($param),)* stochastic_rs_core::simd_rng::Unseeded)); },
        }
        Ok(s)
      }

      fn sample<'py>(&self, py: pyo3::Python<'py>) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
        use numpy::IntoPyArray;
        use $crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        $crate::py_device_dispatch!(self, |inner| {
          let [a, b] = inner.sample();
          (a.into_pyarray(py).into_py_any(py).unwrap(), b.into_pyarray(py).into_py_any(py).unwrap())
        })
      }

      /// Same reproducibility guarantee as `py_process_1d!`'s `sample_par`
      /// (see its doc comment) — bit-identical across rayon thread-pool
      /// sizes for a given seed and `m`.
      fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
        use numpy::IntoPyArray;
        use numpy::ndarray::Array2;
        use $crate::traits::ProcessExt;
        use pyo3::IntoPyObjectExt;
        $crate::py_device_dispatch!(self, |inner| {
          let samples = inner.sample_par(m);
          let n = samples[0][0].len();
          let mut r0 = Array2::zeros((m, n));
          let mut r1 = Array2::zeros((m, n));
          for (i, [a, b]) in samples.iter().enumerate() {
            r0.row_mut(i).assign(a);
            r1.row_mut(i).assign(b);
          }
          (r0.into_pyarray(py).into_py_any(py).unwrap(), r1.into_pyarray(py).into_py_any(py).unwrap())
        })
      }
    }
  };
}

#[cfg(not(feature = "python"))]
#[macro_export]
macro_rules! py_process_2x1d {
  ($($tt:tt)*) => {};
}

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

  // Same class with a `device=` argument (see `python_device::Device`).
  ($py_name:ident, $inner:ident,
    sig: ($($sig:tt)*),
    params: ($($param:ident : $pty:ty),* $(,)?),
    device
  ) => {
    #[pyo3::prelude::pyclass]
    pub struct $py_name {
      inner_f32: Option<$inner<f32>>,
      inner_f64: Option<$inner<f64>>,
      seeded_f32: Option<$inner<f32, stochastic_rs_core::simd_rng::Deterministic>>,
      seeded_f64: Option<$inner<f64, stochastic_rs_core::simd_rng::Deterministic>>,
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
          (Some(sd), "f32") => { s.seeded_f32 = Some($inner::new($(stochastic_rs_core::python::IntoF32::into_f32($param),)* stochastic_rs_core::simd_rng::Deterministic::new(sd))); },
          (Some(sd), _) => { s.seeded_f64 = Some($inner::new($(stochastic_rs_core::python::IntoF64::into_f64($param),)* stochastic_rs_core::simd_rng::Deterministic::new(sd))); },
          (None, "f32") => { s.inner_f32 = Some($inner::new($(stochastic_rs_core::python::IntoF32::into_f32($param),)* stochastic_rs_core::simd_rng::Unseeded)); },
          (None, _) => { s.inner_f64 = Some($inner::new($(stochastic_rs_core::python::IntoF64::into_f64($param),)* stochastic_rs_core::simd_rng::Unseeded)); },
        }
        Ok(s)
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

/// Compile-time backend switch: generates the whole `impl` block whose
/// `on::<B2>()` re-types a process to sample on another
/// [`crate::device::Backend`] with zero runtime cost. The marker params `B`
/// (source) and `B2` (target) are appended automatically — pass the remaining
/// generics in `[..]`, the type without `B`, the moved fields, the storage
/// form, then any trailing `where` bounds.
///
/// Storage form:
/// - `via fgn`     — backend carried through an inner `fgn: Fgn<_, _, B>` field.
/// - `via phantom` — backend carried through a `backend: Cpu<B>` field.
/// - `via host`    — a process with a host sampler only (`B2: HostBackend`), backend carried
///   through its public `backend: Cpu<B>` field.
/// - `via euler`   — a process with Euler-engine device kernels (`B2: EulerBackend`), same field.
macro_rules! backend_switch {
  (
    [$($gen:tt)*] $ty:ident<$t:ident $(, $targ:ident)* $(,)?> { $($field:ident),* $(,)? } via fgn
    $(where $($wc:tt)*)?
  ) => {
    impl<$($gen)*, B> $ty<$t $(, $targ)*, B> $(where $($wc)*)? {
      /// The same process on backend `B2` with that backend's default handle
      /// (the environment's device ordinal and batch budget).
      pub fn on<B2: $crate::device::FgnBackend<$t> + Default>(self) -> $ty<$t $(, $targ)*, B2> {
        self.on_device(B2::default())
      }

      /// The same process on the given backend handle.
      pub fn on_device<B2: $crate::device::FgnBackend<$t>>(self, device: B2) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          fgn: self.fgn.on_device(device),
        }
      }
    }
  };
  (
    [$($gen:tt)*] $ty:ident<$t:ident $(, $targ:ident)* $(,)?> { $($field:ident),* $(,)? } via phantom
    $(where $($wc:tt)*)?
  ) => {
    impl<$($gen)*, B> $ty<$t $(, $targ)*, B> $(where $($wc)*)? {
      /// The same process on backend `B2` with that backend's default handle
      /// (the environment's device ordinal and batch budget).
      pub fn on<B2: $crate::device::FgnBackend<$t> + Default>(self) -> $ty<$t $(, $targ)*, B2> {
        self.on_device(B2::default())
      }

      /// The same process on the given backend handle.
      pub fn on_device<B2: $crate::device::FgnBackend<$t>>(self, device: B2) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          backend: device,
        }
      }
    }
  };
  (
    [$($gen:tt)*] $ty:ident<$t:ident $(, $targ:ident)* $(,)?> { $($field:ident),* $(,)? } via host
    $(where $($wc:tt)*)?
  ) => {
    impl<$($gen)*, B> $ty<$t $(, $targ)*, B> $(where $($wc)*)? {
      /// The same process on backend `B2` with that backend's default handle.
      pub fn on<B2: $crate::device::HostBackend + Default>(self) -> $ty<$t $(, $targ)*, B2> {
        self.on_device(B2::default())
      }

      /// The same process on the given backend handle.
      pub fn on_device<B2: $crate::device::HostBackend>(self, device: B2) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          backend: device,
        }
      }
    }
  };
  (
    [$($gen:tt)*] $ty:ident<$t:ident $(, $targ:ident)* $(,)?> { $($field:ident),* $(,)? } via euler
    $(where $($wc:tt)*)?
  ) => {
    impl<$($gen)*, B> $ty<$t $(, $targ)*, B> $(where $($wc)*)? {
      /// The same process on backend `B2` with that backend's default handle
      /// (the environment's device ordinal and batch budget).
      pub fn on<B2: $crate::euler::EulerBackend<$t> + Default>(self) -> $ty<$t $(, $targ)*, B2> {
        self.on_device(B2::default())
      }

      /// The same process on the given backend handle, e.g.
      /// `CudaNative::new(1).with_batch_budget(256 << 20)`.
      pub fn on_device<B2: $crate::euler::EulerBackend<$t>>(self, device: B2) -> $ty<$t $(, $targ)*, B2> {
        $ty {
          $($field: self.$field,)*
          backend: device,
        }
      }
    }
  };
}
