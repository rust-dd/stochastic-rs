use ndarray::Array0;
use ndarray::Array1;
use ndarray::Axis;
use ndarray::Dim;
use ndarray_rand::RandomExt;
use rand::rng;
use rand_distr::Distribution;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct CustomJt<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub n: Option<usize>,
  pub t_max: Option<T>,
  pub distribution: D,
}

impl<T, D> CustomJt<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(n: Option<usize>, t_max: Option<T>, distribution: D) -> Self {
    CustomJt {
      n,
      t_max,
      distribution,
    }
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyCustomJt {
  inner: CustomJt<f64, crate::traits::CallableDist>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyCustomJt {
  #[new]
  #[pyo3(signature = (distribution, n=None, t_max=None))]
  fn new(distribution: pyo3::Py<pyo3::PyAny>, n: Option<usize>, t_max: Option<f64>) -> Self {
    Self {
      inner: CustomJt::new(n, t_max, crate::traits::CallableDist::new(distribution)),
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    self
      .inner
      .sample()
      .into_pyarray(py)
      .into_py_any(py)
      .unwrap()
  }
}

impl<T, D> ProcessExt<T> for CustomJt<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    if let Some(n) = self.n {
      let random = Array1::random(n, &self.distribution);
      let mut x = Array1::<T>::zeros(n);
      for i in 1..n {
        x[i] = x[i - 1] + random[i - 1];
      }

      x
    } else if let Some(t_max) = self.t_max {
      let mut x = Array1::from(vec![T::zero()]);
      let mut t = T::zero();

      while t < t_max {
        t += self.distribution.sample(&mut rng());
        x.push(Axis(0), Array0::from_elem(Dim(()), t).view())
          .unwrap();
      }

      x
    } else {
      panic!("n or t_max must be provided");
    }
  }
}
