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
  inner: CompoundPoisson<f64, crate::traits::CallableDist>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyCompoundPoisson {
  #[new]
  #[pyo3(signature = (distribution, lambda_, n=None, t_max=None))]
  fn new(distribution: pyo3::Py<pyo3::PyAny>, lambda_: f64, n: Option<usize>, t_max: Option<f64>) -> Self {
    Self {
      inner: CompoundPoisson::new(
        crate::traits::CallableDist::new(distribution),
        Poisson::new(lambda_, n, t_max),
      ),
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
    use numpy::IntoPyArray;
    use crate::traits::ProcessExt;
    use pyo3::IntoPyObjectExt;
    let [p, cum, j] = self.inner.sample();
    (
      p.into_pyarray(py).into_py_any(py).unwrap(),
      cum.into_pyarray(py).into_py_any(py).unwrap(),
      j.into_pyarray(py).into_py_any(py).unwrap(),
    )
  }
}
