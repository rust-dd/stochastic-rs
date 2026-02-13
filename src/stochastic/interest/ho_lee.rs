use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::traits::Fn1D;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

#[allow(non_snake_case)]
pub struct HoLee<T: FloatExt> {
  pub f_T: Option<Fn1D<T>>,
  pub theta: Option<T>,
  pub sigma: T,
  pub n: usize,
  pub t: Option<T>,
  gn: Gn<T>,
}

impl<T: FloatExt> HoLee<T> {
  pub fn new(f_T: Option<Fn1D<T>>, theta: Option<T>, sigma: T, n: usize, t: Option<T>) -> Self {
    assert!(
      theta.is_some() || f_T.is_some(),
      "theta or f_T must be provided"
    );

    Self {
      f_T,
      theta,
      sigma,
      n,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for HoLee<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = &self.gn.sample();

    let mut r = Array1::<T>::zeros(self.n);

    for i in 1..self.n {
      let t = T::from_usize_(i) * dt;
      let drift = if let Some(ref f) = self.f_T {
        f.call(t) + self.sigma.powf(T::from_usize_(2)) * t
      } else {
        self.theta.unwrap() + self.sigma.powf(T::from_usize_(2)) * t
      };

      r[i] = r[i - 1] + drift * dt + self.sigma * gn[i - 1];
    }

    r
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyHoLee {
  inner: HoLee<f64>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyHoLee {
  #[new]
  #[pyo3(signature = (sigma, n, f_T=None, theta=None, t=None))]
  fn new(
    sigma: f64, n: usize,
    f_T: Option<pyo3::Py<pyo3::PyAny>>, theta: Option<f64>, t: Option<f64>,
  ) -> Self {
    Self {
      inner: HoLee::new(f_T.map(Fn1D::Py), theta, sigma, n, t),
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use crate::traits::ProcessExt;
    use pyo3::IntoPyObjectExt;
    self.inner.sample().into_pyarray(py).into_py_any(py).unwrap()
  }
}
