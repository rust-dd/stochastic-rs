use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::Fn2D;
use crate::traits::ProcessExt;

pub struct HJM<T: FloatExt> {
  pub a: Fn1D<T>,
  pub b: Fn1D<T>,
  pub p: Fn2D<T>,
  pub q: Fn2D<T>,
  pub v: Fn2D<T>,
  pub alpha: Fn2D<T>,
  pub sigma: Fn2D<T>,
  pub n: usize,
  pub r0: Option<T>,
  pub p0: Option<T>,
  pub f0: Option<T>,
  pub t: Option<T>,
  gn: Gn<T>,
}

impl<T: FloatExt> HJM<T> {
  pub fn new(
    a: impl Into<Fn1D<T>>,
    b: impl Into<Fn1D<T>>,
    p: impl Into<Fn2D<T>>,
    q: impl Into<Fn2D<T>>,
    v: impl Into<Fn2D<T>>,
    alpha: impl Into<Fn2D<T>>,
    sigma: impl Into<Fn2D<T>>,
    n: usize,
    r0: Option<T>,
    p0: Option<T>,
    f0: Option<T>,
    t: Option<T>,
  ) -> Self {
    Self {
      a: a.into(),
      b: b.into(),
      p: p.into(),
      q: q.into(),
      v: v.into(),
      alpha: alpha.into(),
      sigma: sigma.into(),
      n,
      r0,
      p0,
      f0,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for HJM<T> {
  type Output = [Array1<T>; 3];

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();

    let mut r = Array1::<T>::zeros(self.n);
    let mut p = Array1::<T>::zeros(self.n);
    let mut f_ = Array1::<T>::zeros(self.n);

    r[0] = self.r0.unwrap_or(T::zero());
    p[0] = self.p0.unwrap_or(T::zero());
    f_[0] = self.f0.unwrap_or(T::zero());

    let gn1 = &self.gn.sample();
    let gn2 = &self.gn.sample();
    let gn3 = &self.gn.sample();

    let t_max = self.t.unwrap_or(T::one());

    for i in 1..self.n {
      let t = T::from_usize_(i) * dt;

      r[i] = r[i - 1] + self.a.call(t) * dt + self.b.call(t) * gn1[i - 1];
      p[i] = p[i - 1]
        + self.p.call(t, t_max) * (self.q.call(t, t_max) * dt + self.v.call(t, t_max) * gn2[i - 1]);
      f_[i] = f_[i - 1] + self.alpha.call(t, t_max) * dt + self.sigma.call(t, t_max) * gn3[i - 1];
    }

    [r, p, f_]
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyHJM {
  inner: HJM<f64>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyHJM {
  #[new]
  #[pyo3(signature = (a, b, p, q, v, alpha, sigma, n, r0=None, p0=None, f0=None, t=None))]
  fn new(
    a: pyo3::Py<pyo3::PyAny>,
    b: pyo3::Py<pyo3::PyAny>,
    p: pyo3::Py<pyo3::PyAny>,
    q: pyo3::Py<pyo3::PyAny>,
    v: pyo3::Py<pyo3::PyAny>,
    alpha: pyo3::Py<pyo3::PyAny>,
    sigma: pyo3::Py<pyo3::PyAny>,
    n: usize,
    r0: Option<f64>,
    p0: Option<f64>,
    f0: Option<f64>,
    t: Option<f64>,
  ) -> Self {
    use crate::traits::Fn2D;
    Self {
      inner: HJM::new(
        Fn1D::Py(a),
        Fn1D::Py(b),
        Fn2D::Py(p),
        Fn2D::Py(q),
        Fn2D::Py(v),
        Fn2D::Py(alpha),
        Fn2D::Py(sigma),
        n,
        r0,
        p0,
        f0,
        t,
      ),
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
    let [r, p, f] = self.inner.sample();
    (
      r.into_pyarray(py).into_py_any(py).unwrap(),
      p.into_pyarray(py).into_py_any(py).unwrap(),
      f.into_pyarray(py).into_py_any(py).unwrap(),
    )
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn zero_1d(_: f64) -> f64 {
    0.0
  }

  fn zero_2d(_: f64, _: f64) -> f64 {
    0.0
  }

  fn one_2d(_: f64, _: f64) -> f64 {
    1.0
  }

  fn tmax_2d(_: f64, t_max: f64) -> f64 {
    t_max
  }

  #[test]
  fn default_t_max_is_one() {
    let model = HJM::new(
      zero_1d as fn(f64) -> f64,
      zero_1d as fn(f64) -> f64,
      tmax_2d as fn(f64, f64) -> f64,
      one_2d as fn(f64, f64) -> f64,
      zero_2d as fn(f64, f64) -> f64,
      zero_2d as fn(f64, f64) -> f64,
      zero_2d as fn(f64, f64) -> f64,
      3,
      Some(0.0),
      Some(0.0),
      Some(0.0),
      None,
    );

    let [_r, p, _f] = model.sample();
    assert!((p[1] - 0.5).abs() < 1e-12);
    assert!((p[2] - 1.0).abs() < 1e-12);
  }
}
