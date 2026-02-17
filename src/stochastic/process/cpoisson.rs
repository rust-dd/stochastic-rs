use ndarray::Array1;
use ndarray::Axis;
use rand::rng;
use rand_distr::Distribution;

use super::poisson::Poisson;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct CompoundPoisson<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub distribution: D,
  pub poisson: Poisson<T>,
}

impl<T, D> CompoundPoisson<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(distribution: D, poisson: Poisson<T>) -> Self {
    Self {
      distribution,
      poisson,
    }
  }
}

impl<T, D> ProcessExt<T> for CompoundPoisson<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = [Array1<T>; 3];

  fn sample(&self) -> Self::Output {
    let poisson = self.poisson.sample();
    let mut jumps = Array1::<T>::zeros(poisson.len());
    for i in 1..poisson.len() {
      jumps[i] = self.distribution.sample(&mut rng());
    }

    let mut cum_jupms = jumps.clone();
    cum_jupms.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

    [poisson, cum_jupms, jumps]
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyCompoundPoisson {
  inner_f32: Option<CompoundPoisson<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<CompoundPoisson<f64, crate::traits::CallableDist<f64>>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyCompoundPoisson {
  #[new]
  #[pyo3(signature = (distribution, lambda_, n=None, t_max=None, dtype=None))]
  fn new(
    distribution: pyo3::Py<pyo3::PyAny>,
    lambda_: f64,
    n: Option<usize>,
    t_max: Option<f64>,
    dtype: Option<&str>,
  ) -> Self {
    match dtype.unwrap_or("f64") {
      "f32" => Self {
        inner_f32: Some(CompoundPoisson::new(
          crate::traits::CallableDist::new(distribution),
          Poisson::new(lambda_ as f32, n, t_max.map(|v| v as f32)),
        )),
        inner_f64: None,
      },
      _ => Self {
        inner_f32: None,
        inner_f64: Some(CompoundPoisson::new(
          crate::traits::CallableDist::new(distribution),
          Poisson::new(lambda_, n, t_max),
        )),
      },
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
