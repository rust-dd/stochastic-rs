use ndarray::Array1;
use rand_distr::Distribution;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::process::cpoisson::CompoundPoisson;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct LevyDiffusion<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub gamma: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub cpoisson: CompoundPoisson<T, D>,
  gn: Gn<T>,
}

impl<T, D> LevyDiffusion<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(
    gamma: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    cpoisson: CompoundPoisson<T, D>,
  ) -> Self {
    Self {
      gamma,
      sigma,
      n,
      x0,
      t,
      cpoisson,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T, D> ProcessExt<T> for LevyDiffusion<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = &self.gn.sample();

    let mut levy = Array1::<T>::zeros(self.n);
    levy[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      let [.., jumps] = self.cpoisson.sample();
      levy[i] = levy[i - 1] + self.gamma * dt + self.sigma * gn[i - 1] + jumps.sum();
    }

    levy
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyLevyDiffusion {
  inner_f32: Option<LevyDiffusion<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<LevyDiffusion<f64, crate::traits::CallableDist<f64>>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyLevyDiffusion {
  #[new]
  #[pyo3(signature = (gamma_, sigma, distribution, lambda_, n, x0=None, t=None, dtype=None))]
  fn new(
    gamma_: f64,
    sigma: f64,
    distribution: pyo3::Py<pyo3::PyAny>,
    lambda_: f64,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    dtype: Option<&str>,
  ) -> Self {
    use crate::stochastic::process::poisson::Poisson;
    match dtype.unwrap_or("f64") {
      "f32" => {
        let cpoisson = CompoundPoisson::new(
          crate::traits::CallableDist::new(distribution),
          Poisson::new(lambda_ as f32, Some(n), t.map(|v| v as f32)),
        );
        Self {
          inner_f32: Some(LevyDiffusion::new(
            gamma_ as f32, sigma as f32, n,
            x0.map(|v| v as f32), t.map(|v| v as f32), cpoisson,
          )),
          inner_f64: None,
        }
      }
      _ => {
        let cpoisson = CompoundPoisson::new(
          crate::traits::CallableDist::new(distribution),
          Poisson::new(lambda_, Some(n), t),
        );
        Self {
          inner_f32: None,
          inner_f64: Some(LevyDiffusion::new(gamma_, sigma, n, x0, t, cpoisson)),
        }
      }
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
