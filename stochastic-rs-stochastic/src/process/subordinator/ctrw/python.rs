use super::*;

#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyCtrw {
  inner_f32: Option<Ctrw<f32>>,
  inner_f64: Option<Ctrw<f64>>,
}

#[pyo3::prelude::pymethods]
impl PyCtrw {
  #[new]
  #[pyo3(signature = (
    waiting_law,
    waiting_p1,
    waiting_p2=None,
    jump_law="normal",
    jump_p1=0.0,
    jump_p2=None,
    n=1000,
    x0=None,
    t=None,
    dtype=None
  ))]
  fn new(
    waiting_law: &str,
    waiting_p1: f64,
    waiting_p2: Option<f64>,
    jump_law: &str,
    jump_p1: f64,
    jump_p2: Option<f64>,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    dtype: Option<&str>,
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| {
      let waiting_f64 = match waiting_law.to_ascii_lowercase().as_str() {
        "exp" | "exponential" => CtrwWaitingLaw::Exponential { rate: waiting_p1 },
        "gamma" => CtrwWaitingLaw::Gamma {
          shape: waiting_p1,
          rate: waiting_p2.unwrap_or(1.0),
        },
        "ig" | "inverse_gaussian" | "inversegaussian" => CtrwWaitingLaw::InverseGaussian {
          mu: waiting_p1,
          lambda: waiting_p2.unwrap_or(1.0),
        },
        "stable" | "positive_stable" | "positivestable" => CtrwWaitingLaw::PositiveStable {
          alpha: waiting_p1,
          scale: waiting_p2.unwrap_or(1.0),
        },
        _ => panic!(
          "PyCtrw: invalid waiting_law '{}' — expected one of 'exponential' | 'gamma' | 'inverse_gaussian' | 'inversegaussian' | 'ig' | 'stable' | 'positive_stable' | 'positivestable'",
          waiting_law
        ),
      };

      let jumps_f64 = match jump_law.to_ascii_lowercase().as_str() {
        "normal" => CtrwJumpLaw::Normal {
          mean: jump_p1,
          std: jump_p2.unwrap_or(1.0),
        },
        "stable" | "symmetric_stable" | "symmetricstable" => CtrwJumpLaw::SymmetricStable {
          alpha: jump_p1,
          scale: jump_p2.unwrap_or(1.0),
        },
        "rademacher" => CtrwJumpLaw::Rademacher {
          scale: jump_p1.abs(),
        },
        _ => panic!(
          "PyCtrw: invalid jump_law '{}' — expected one of 'normal' | 'symmetric_stable' | 'symmetricstable' | 'stable' | 'rademacher'",
          jump_law
        ),
      };

      match dtype.unwrap_or("f64") {
        "f32" => {
          let waiting_f32 = match waiting_f64 {
            CtrwWaitingLaw::Exponential { rate } => {
              CtrwWaitingLaw::Exponential { rate: rate as f32 }
            }
            CtrwWaitingLaw::Gamma { shape, rate } => CtrwWaitingLaw::Gamma {
              shape: shape as f32,
              rate: rate as f32,
            },
            CtrwWaitingLaw::InverseGaussian { mu, lambda } => CtrwWaitingLaw::InverseGaussian {
              mu: mu as f32,
              lambda: lambda as f32,
            },
            CtrwWaitingLaw::PositiveStable { alpha, scale } => CtrwWaitingLaw::PositiveStable {
              alpha: alpha as f32,
              scale: scale as f32,
            },
          };
          let jumps_f32 = match jumps_f64 {
            CtrwJumpLaw::Normal { mean, std } => CtrwJumpLaw::Normal {
              mean: mean as f32,
              std: std as f32,
            },
            CtrwJumpLaw::SymmetricStable { alpha, scale } => CtrwJumpLaw::SymmetricStable {
              alpha: alpha as f32,
              scale: scale as f32,
            },
            CtrwJumpLaw::Rademacher { scale } => CtrwJumpLaw::Rademacher {
              scale: scale as f32,
            },
          };
          Self {
            inner_f32: Some(Ctrw::new(
              waiting_f32,
              jumps_f32,
              n,
              x0.map(|v| v as f32),
              t.map(|v| v as f32),
              Unseeded,
            )),
            inner_f64: None,
          }
        }
        _ => Self {
          inner_f32: None,
          inner_f64: Some(Ctrw::new(waiting_f64, jumps_f64, n, x0, t, Unseeded)),
        },
      }
    })
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;

      if let Some(ref inner) = self.inner_f64 {
        inner.sample().into_pyarray(py).into_py_any(py).unwrap()
      } else if let Some(ref inner) = self.inner_f32 {
        inner.sample().into_pyarray(py).into_py_any(py).unwrap()
      } else {
        unreachable!()
      }
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

      if let Some(ref inner) = self.inner_f64 {
        let paths = inner.sample_par(m);
        let n = paths[0].len();
        let mut result = Array2::<f64>::zeros((m, n));
        for (i, path) in paths.iter().enumerate() {
          result.row_mut(i).assign(path);
        }
        result.into_pyarray(py).into_py_any(py).unwrap()
      } else if let Some(ref inner) = self.inner_f32 {
        let paths = inner.sample_par(m);
        let n = paths[0].len();
        let mut result = Array2::<f32>::zeros((m, n));
        for (i, path) in paths.iter().enumerate() {
          result.row_mut(i).assign(path);
        }
        result.into_pyarray(py).into_py_any(py).unwrap()
      } else {
        unreachable!()
      }
    })
  }
}
