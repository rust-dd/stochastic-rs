//! # Merton
//!
//! $$
//! \frac{dS_t}{S_{t^-}}=(\mu-\lambda\kappa)dt+\sigma dW_t+(Y-1)dN_t
//! $$
//!
use ndarray::Array1;
use rand_distr::Distribution;

use crate::stochastic::process::cpoisson::CompoundPoisson;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct Merton<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub alpha: T,
  pub sigma: T,
  pub lambda: T,
  pub theta: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub cpoisson: CompoundPoisson<T, D>,
}

impl<T, D> Merton<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(
    alpha: T,
    sigma: T,
    lambda: T,
    theta: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    cpoisson: CompoundPoisson<T, D>,
  ) -> Self {
    Self {
      alpha,
      sigma,
      lambda,
      theta,
      n,
      x0,
      t,
      cpoisson,
    }
  }
}

impl<T, D> ProcessExt<T> for Merton<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = if self.n > 1 {
      self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
    } else {
      T::zero()
    };
    let jump_increments = self.cpoisson.sample_grid_increments(self.n, dt);
    let mut gn = Array1::<T>::zeros(self.n.saturating_sub(1));
    if let Some(gn_slice) = gn.as_slice_mut() {
      let sqrt_dt = dt.sqrt();
      T::fill_standard_normal_scaled_slice(gn_slice, sqrt_dt);
    }

    let mut merton = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return merton;
    }
    merton[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      merton[i] = merton[i - 1]
        + (self.alpha
          - self.sigma.powf(T::from_usize(2).unwrap()) / T::from_usize(2).unwrap()
          - self.lambda * self.theta)
          * dt
        + self.sigma * gn[i - 1]
        + jump_increments[i];
    }

    merton
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyMerton {
  inner_f32: Option<Merton<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<Merton<f64, crate::traits::CallableDist<f64>>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyMerton {
  #[new]
  #[pyo3(signature = (alpha, sigma, lambda_, theta, distribution, n, x0=None, t=None, dtype=None))]
  fn new(
    alpha: f64,
    sigma: f64,
    lambda_: f64,
    theta: f64,
    distribution: pyo3::Py<pyo3::PyAny>,
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
          inner_f32: Some(Merton::new(
            alpha as f32,
            sigma as f32,
            lambda_ as f32,
            theta as f32,
            n,
            x0.map(|v| v as f32),
            t.map(|v| v as f32),
            cpoisson,
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
          inner_f64: Some(Merton::new(
            alpha, sigma, lambda_, theta, n, x0, t, cpoisson,
          )),
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
