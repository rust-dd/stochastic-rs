//! Type-erased `Fn1D` / `Fn2D` callables and the Python-feature-gated
//! `CallableDist` adapter.

use super::float::FloatExt;

pub enum Fn1D<T: FloatExt> {
  Native(fn(T) -> T),
  #[cfg(feature = "python")]
  Py(pyo3::Py<pyo3::PyAny>),
}

impl<T: FloatExt> Fn1D<T> {
  pub fn call(&self, t: T) -> T {
    match self {
      Fn1D::Native(f) => f(t),
      #[cfg(feature = "python")]
      Fn1D::Py(callable) => pyo3::Python::attach(|py| {
        let result: f64 = callable
          .call1(py, (t.to_f64().unwrap(),))
          .unwrap()
          .extract(py)
          .unwrap();
        T::from_f64_fast(result)
      }),
    }
  }
}

impl<T: FloatExt> From<fn(T) -> T> for Fn1D<T> {
  fn from(f: fn(T) -> T) -> Self {
    Fn1D::Native(f)
  }
}

pub enum Fn2D<T: FloatExt> {
  Native(fn(T, T) -> T),
  #[cfg(feature = "python")]
  Py(pyo3::Py<pyo3::PyAny>),
}

impl<T: FloatExt> Fn2D<T> {
  pub fn call(&self, t: T, u: T) -> T {
    match self {
      Fn2D::Native(f) => f(t, u),
      #[cfg(feature = "python")]
      Fn2D::Py(callable) => pyo3::Python::attach(|py| {
        let result: f64 = callable
          .call1(py, (t.to_f64().unwrap(), u.to_f64().unwrap()))
          .unwrap()
          .extract(py)
          .unwrap();
        T::from_f64_fast(result)
      }),
    }
  }
}

impl<T: FloatExt> From<fn(T, T) -> T> for Fn2D<T> {
  fn from(f: fn(T, T) -> T) -> Self {
    Fn2D::Native(f)
  }
}

#[cfg(feature = "python")]
pub struct CallableDist<T: FloatExt> {
  callable: pyo3::Py<pyo3::PyAny>,
  _phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "python")]
impl<T: FloatExt> CallableDist<T> {
  pub fn new(callable: pyo3::Py<pyo3::PyAny>) -> Self {
    Self {
      callable,
      _phantom: std::marker::PhantomData,
    }
  }
}

#[cfg(feature = "python")]
impl<T: FloatExt> rand_distr::Distribution<T> for CallableDist<T> {
  fn sample<R: rand::Rng + ?Sized>(&self, _rng: &mut R) -> T {
    pyo3::Python::attach(|py| {
      let result: f64 = self.callable.call0(py).unwrap().extract::<f64>(py).unwrap();
      T::from_f64_fast(result)
    })
  }
}
