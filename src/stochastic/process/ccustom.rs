//! # Ccustom
//!
//! $$
//! dX_t=a(t,X_t)dt+b(t,X_t)dW_t+\sum_{k=1}^{dN_t}J_k
//! $$
//!
use ndarray::Array1;
use ndarray::Axis;
use rand::rng;
use rand_distr::Distribution;

use super::customjt::CustomJt;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct CompoundCustom<T, D1, D2>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync,
  D2: Distribution<T> + Send + Sync,
{
  /// Optional fixed number of generated events.
  pub n: Option<usize>,
  /// Optional horizon for time-based generation.
  /// Used when `n` is `None`.
  pub t_max: Option<T>,
  /// Distribution of jump magnitudes.
  pub jumps_distribution: D1,
  /// Distribution of jump waiting times / event times.
  pub jump_times_distribution: D2,
  /// Underlying jump-time generator used internally.
  pub customjt: CustomJt<T, D2>,
}

impl<T, D1, D2> CompoundCustom<T, D1, D2>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync,
  D2: Distribution<T> + Send + Sync,
{
  pub fn new(
    n: Option<usize>,
    t_max: Option<T>,
    jumps_distribution: D1,
    jump_times_distribution: D2,
    customjt: CustomJt<T, D2>,
  ) -> Self {
    if n.is_none() && t_max.is_none() {
      panic!("n or t_max must be provided");
    }

    Self {
      n,
      t_max,
      jumps_distribution,
      jump_times_distribution,
      customjt,
    }
  }
}

impl<T, D1, D2> ProcessExt<T> for CompoundCustom<T, D1, D2>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync,
  D2: Distribution<T> + Send + Sync,
{
  type Output = [Array1<T>; 3];

  fn sample(&self) -> Self::Output {
    let p = self.customjt.sample();
    let mut jumps = Array1::<T>::zeros(self.n.unwrap_or(p.len()));
    let mut rng = rng();
    for i in 1..p.len() {
      jumps[i] = self.jumps_distribution.sample(&mut rng);
    }

    let mut cum_jupms = jumps.clone();
    cum_jupms.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

    [p, cum_jupms, jumps]
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyCompoundCustom {
  inner_f32:
    Option<CompoundCustom<f32, crate::traits::CallableDist<f32>, crate::traits::CallableDist<f32>>>,
  inner_f64:
    Option<CompoundCustom<f64, crate::traits::CallableDist<f64>, crate::traits::CallableDist<f64>>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyCompoundCustom {
  #[new]
  #[pyo3(signature = (jumps_distribution, jump_times_distribution, n=None, t_max=None, dtype=None))]
  fn new(
    jumps_distribution: pyo3::Py<pyo3::PyAny>,
    jump_times_distribution: pyo3::Py<pyo3::PyAny>,
    n: Option<usize>,
    t_max: Option<f64>,
    dtype: Option<&str>,
  ) -> Self {
    match dtype.unwrap_or("f64") {
      "f32" => {
        let (jt_dist, customjt_dist) = pyo3::Python::attach(|py| {
          let a = jump_times_distribution.clone_ref(py);
          let b = jump_times_distribution;
          (
            crate::traits::CallableDist::<f32>::new(a),
            crate::traits::CallableDist::<f32>::new(b),
          )
        });
        let customjt = CustomJt::new(n, t_max.map(|v| v as f32), customjt_dist);
        Self {
          inner_f32: Some(CompoundCustom::new(
            n,
            t_max.map(|v| v as f32),
            crate::traits::CallableDist::new(jumps_distribution),
            jt_dist,
            customjt,
          )),
          inner_f64: None,
        }
      }
      _ => {
        let (jt_dist, customjt_dist) = pyo3::Python::attach(|py| {
          let a = jump_times_distribution.clone_ref(py);
          let b = jump_times_distribution;
          (
            crate::traits::CallableDist::<f64>::new(a),
            crate::traits::CallableDist::<f64>::new(b),
          )
        });
        let customjt = CustomJt::new(n, t_max, customjt_dist);
        Self {
          inner_f32: None,
          inner_f64: Some(CompoundCustom::new(
            n,
            t_max,
            crate::traits::CallableDist::new(jumps_distribution),
            jt_dist,
            customjt,
          )),
        }
      }
    }
  }

  fn sample<'py>(
    &self,
    py: pyo3::Python<'py>,
  ) -> (
    pyo3::Py<pyo3::PyAny>,
    pyo3::Py<pyo3::PyAny>,
    pyo3::Py<pyo3::PyAny>,
  ) {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    if let Some(ref inner) = self.inner_f64 {
      let [p, cum, j] = inner.sample();
      (
        p.into_pyarray(py).into_py_any(py).unwrap(),
        cum.into_pyarray(py).into_py_any(py).unwrap(),
        j.into_pyarray(py).into_py_any(py).unwrap(),
      )
    } else if let Some(ref inner) = self.inner_f32 {
      let [p, cum, j] = inner.sample();
      (
        p.into_pyarray(py).into_py_any(py).unwrap(),
        cum.into_pyarray(py).into_py_any(py).unwrap(),
        j.into_pyarray(py).into_py_any(py).unwrap(),
      )
    } else {
      unreachable!()
    }
  }
}
