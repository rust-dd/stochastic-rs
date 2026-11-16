use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

// Hand-written: `new` takes a `VolterraKernelSpec`, not the flat list `py_process_1d!` expands.
#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyVolterra {
  inner: Option<Volterra<f64>>,
  seeded: Option<Volterra<f64, Deterministic>>,
}

#[pyo3::prelude::pymethods]
impl PyVolterra {
  /// Build a Volterra process.
  ///
  /// # Arguments
  /// * `kernel` — `"fbm"`, `"power_law"` (alias `"powerlaw"`), or `"exponential"`.
  /// * `param` — the kernel's scalar parameter: Hurst $H$ for `"fbm"`,
  ///   exponent $\gamma$ for `"power_law"`, decay $\beta$ for `"exponential"`.
  /// * `n` — number of grid points.
  /// * `t` — time horizon (default $1$).
  /// * `seed` — optional u64 seed for reproducibility.
  #[new]
  #[pyo3(signature = (kernel, param, n, t = None, seed = None))]
  fn new(
    kernel: &str,
    param: f64,
    n: usize,
    t: Option<f64>,
    seed: Option<u64>,
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| {
      let kernel = match kernel.to_ascii_lowercase().as_str() {
        "fbm" | "fractional_bm" | "fractionalbm" => VolterraKernelSpec::FractionalBM { h: param },
        "power_law" | "powerlaw" => VolterraKernelSpec::PowerLaw { gamma: param },
        "exponential" | "exp" => VolterraKernelSpec::Exponential { beta: param },
        other => {
          panic!(
            "PyVolterra: unknown kernel '{other}' — expected one of 'fbm' | 'fractional_bm' | 'fractionalbm' | 'power_law' | 'powerlaw' | 'exponential' | 'exp'"
          )
        }
      };
      match seed {
        Some(sd) => Self {
          inner: None,
          seeded: Some(Volterra::<f64, Deterministic>::new(
            kernel,
            n,
            t,
            Deterministic::new(sd),
          )),
        },
        None => Self {
          inner: Some(Volterra::<f64>::new(kernel, n, t, Unseeded)),
          seeded: None,
        },
      }
    })
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      crate::py_dispatch_f64!(self, |inner| inner
        .sample()
        .into_pyarray(py)
        .into_py_any(py)
        .unwrap())
    })
  }

  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use numpy::ndarray::Array2;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      crate::py_dispatch_f64!(self, |inner| {
        let paths = inner.sample_par(m);
        let n = paths[0].len();
        let mut result = Array2::zeros((m, n));
        for (i, path) in paths.iter().enumerate() {
          result.row_mut(i).assign(path);
        }
        result.into_pyarray(py).into_py_any(py).unwrap()
      })
    })
  }
}
