use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::ProcessExt;

pub struct HullWhite<T: FloatExt> {
  pub theta: Fn1D<T>,
  pub alpha: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  gn: Gn<T>,
}

impl<T: FloatExt> HullWhite<T> {
  pub fn new(
    theta: impl Into<Fn1D<T>>,
    alpha: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
  ) -> Self {
    Self {
      theta: theta.into(),
      alpha,
      sigma,
      n,
      x0,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for HullWhite<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = &self.gn.sample();

    let mut hw = Array1::<T>::zeros(self.n);
    hw[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      hw[i] = hw[i - 1]
        + (self.theta.call(T::from_usize_(i) * dt) - self.alpha * hw[i - 1]) * dt
        + self.sigma * gn[i - 1]
    }

    hw
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyHullWhite {
  inner: HullWhite<f64>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyHullWhite {
  #[new]
  #[pyo3(signature = (theta, alpha, sigma, n, x0=None, t=None))]
  fn new(
    theta: pyo3::Py<pyo3::PyAny>,
    alpha: f64,
    sigma: f64,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
  ) -> Self {
    Self {
      inner: HullWhite::new(Fn1D::Py(theta), alpha, sigma, n, x0, t),
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
