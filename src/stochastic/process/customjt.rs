//! # Customjt
//!
//! $$
//! dX_t=a(t,X_t)dt+b(t,X_t)dW_t+\sum_{k=1}^{dN_t}J_k
//! $$
//!
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
  /// Optional fixed number of generated events.
  pub n: Option<usize>,
  /// Optional horizon for time-based generation.
  /// Used when `n` is `None`.
  pub t_max: Option<T>,
  /// Distribution used for generated increments / inter-arrival draws.
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
  inner_f32: Option<CustomJt<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<CustomJt<f64, crate::traits::CallableDist<f64>>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyCustomJt {
  #[new]
  #[pyo3(signature = (distribution, n=None, t_max=None, dtype=None))]
  fn new(
    distribution: pyo3::Py<pyo3::PyAny>,
    n: Option<usize>,
    t_max: Option<f64>,
    dtype: Option<&str>,
  ) -> Self {
    match dtype.unwrap_or("f64") {
      "f32" => Self {
        inner_f32: Some(CustomJt::new(
          n,
          t_max.map(|v| v as f32),
          crate::traits::CallableDist::new(distribution),
        )),
        inner_f64: None,
      },
      _ => Self {
        inner_f32: None,
        inner_f64: Some(CustomJt::new(
          n,
          t_max,
          crate::traits::CallableDist::new(distribution),
        )),
      },
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
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
  }

  fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
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
