//! # Hull White 2f
//!
//! $$
//! dr_t=x_t+y_t,\ dx_t=-a x_t dt+\sigma_1 dW_t^1,\ dy_t=-b y_t dt+\sigma_2 dW_t^2
//! $$
//!
use ndarray::Array1;

use crate::stochastic::noise::cgns::CGNS;
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::ProcessExt;

pub struct HullWhite2F<T: FloatExt> {
  /// Jump-size adjustment / shape parameter.
  pub k: Fn1D<T>,
  /// Long-run target level / model location parameter.
  pub theta: T,
  /// Diffusion/noise scale for factor 1.
  pub sigma1: T,
  /// Diffusion/noise scale for factor 2.
  pub sigma2: T,
  /// Instantaneous correlation parameter.
  pub rho: T,
  /// Model coefficient / user-supplied diffusion term.
  pub b: T,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  cgns: CGNS<T>,
}

impl<T: FloatExt> HullWhite2F<T> {
  pub fn new(
    k: impl Into<Fn1D<T>>,
    theta: T,
    sigma1: T,
    sigma2: T,
    rho: T,
    b: T,
    x0: Option<T>,
    t: Option<T>,
    n: usize,
  ) -> Self {
    Self {
      k: k.into(),
      theta,
      sigma1,
      sigma2,
      rho,
      b,
      x0,
      t,
      n,
      cgns: CGNS::new(rho, n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for HullWhite2F<T> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let dt = self.cgns.dt();
    let [cgn1, cgn2] = &self.cgns.sample();

    let mut x = Array1::<T>::zeros(self.n);
    let mut u = Array1::<T>::zeros(self.n);

    x[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      x[i] = x[i - 1]
        + (self.k.call(T::from_usize_(i) * dt) + u[i - 1] - self.theta * x[i - 1]) * dt
        + self.sigma1 * cgn1[i - 1];

      u[i] = u[i - 1] - self.b * u[i - 1] * dt + self.sigma2 * cgn2[i - 1];
    }

    [x, u]
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyHullWhite2F {
  inner: HullWhite2F<f64>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyHullWhite2F {
  #[new]
  #[pyo3(signature = (k, theta, sigma1, sigma2, rho, b, n, x0=None, t=None))]
  fn new(
    k: pyo3::Py<pyo3::PyAny>,
    theta: f64,
    sigma1: f64,
    sigma2: f64,
    rho: f64,
    b: f64,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
  ) -> Self {
    Self {
      inner: HullWhite2F::new(Fn1D::Py(k), theta, sigma1, sigma2, rho, b, x0, t, n),
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    let [a, b] = self.inner.sample();
    (
      a.into_pyarray(py).into_py_any(py).unwrap(),
      b.into_pyarray(py).into_py_any(py).unwrap(),
    )
  }
}
