use ndarray::Array1;
use ndarray::Array2;

use crate::stochastic::noise::gn::Gn;
use crate::traits::Fn1D;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct ADG<T: FloatExt> {
  pub k: Fn1D<T>,
  pub theta: Fn1D<T>,
  pub sigma: Array1<T>,
  pub phi: Fn1D<T>,
  pub b: Fn1D<T>,
  pub c: Fn1D<T>,
  pub n: usize,
  pub xn: usize,
  pub x0: Array1<T>,
  pub t: Option<T>,
  gn: Gn<T>,
}

impl<T: FloatExt> ADG<T> {
  pub fn new(
    k: impl Into<Fn1D<T>>,
    theta: impl Into<Fn1D<T>>,
    sigma: Array1<T>,
    phi: impl Into<Fn1D<T>>,
    b: impl Into<Fn1D<T>>,
    c: impl Into<Fn1D<T>>,
    n: usize,
    xn: usize,
    x0: Array1<T>,
    t: Option<T>,
  ) -> Self {
    Self {
      k: k.into(),
      theta: theta.into(),
      sigma,
      phi: phi.into(),
      b: b.into(),
      c: c.into(),
      n,
      xn,
      x0,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for ADG<T> {
  type Output = Array2<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();

    let mut adg = Array2::<T>::zeros((self.xn, self.n));
    for i in 0..self.xn {
      adg[(i, 0)] = self.x0[i];
    }

    for i in 0..self.xn {
      let gn = &self.gn.sample();

      for j in 1..self.n {
        let t = T::from_usize_(j) * dt;
        adg[(i, j)] = adg[(i, j - 1)]
          + (self.k.call(t) - self.theta.call(t) * adg[(i, j - 1)]) * dt
          + self.sigma[i] * gn[j - 1];
      }
    }

    let mut r = Array2::zeros((self.xn, self.n));

    for i in 0..self.xn {
      let phi = Array1::<T>::from_shape_fn(self.n, |j| self.phi.call(T::from_usize_(j) * dt));
      let b = Array1::<T>::from_shape_fn(self.n, |j| self.b.call(T::from_usize_(j) * dt));
      let c = Array1::<T>::from_shape_fn(self.n, |j| self.c.call(T::from_usize_(j) * dt));

      let xi = adg.row(i).to_owned();
      let xi_sq = &xi * &xi;
      r.row_mut(i).assign(&(phi + b * &xi + c * xi_sq));
    }

    r
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyADG {
  inner: ADG<f64>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyADG {
  #[new]
  #[pyo3(signature = (k, theta, sigma, phi, b, c, n, xn, x0, t=None))]
  fn new(
    k: pyo3::Py<pyo3::PyAny>, theta: pyo3::Py<pyo3::PyAny>,
    sigma: Vec<f64>,
    phi: pyo3::Py<pyo3::PyAny>, b: pyo3::Py<pyo3::PyAny>, c: pyo3::Py<pyo3::PyAny>,
    n: usize, xn: usize, x0: Vec<f64>, t: Option<f64>,
  ) -> Self {
    Self {
      inner: ADG::new(
        Fn1D::Py(k), Fn1D::Py(theta),
        ndarray::Array1::from_vec(sigma),
        Fn1D::Py(phi), Fn1D::Py(b), Fn1D::Py(c),
        n, xn, ndarray::Array1::from_vec(x0), t,
      ),
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use crate::traits::ProcessExt;
    use pyo3::IntoPyObjectExt;
    self.inner.sample().into_pyarray(py).into_py_any(py).unwrap()
  }
}
