//! # Hull White
//!
//! $$
//! dr_t=\left(\theta(t)-a r_t\right)dt+\sigma dW_t
//! $$
//!
use ndarray::Array1;
use ndarray::s;

use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::ProcessExt;

pub struct HullWhite<T: FloatExt> {
  /// Long-run target level / model location parameter.
  pub theta: Fn1D<T>,
  /// Model shape / loading parameter.
  pub alpha: T,
  /// Diffusion / noise scale parameter.
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
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
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for HullWhite<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut hw = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return hw;
    }

    hw[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return hw;
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let diff_scale = self.sigma * dt.sqrt();
    let mut prev = hw[0];
    let mut tail_view = hw.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("HullWhite output tail must be contiguous");
    T::fill_standard_normal_slice(tail);

    for (k, z) in tail.iter_mut().enumerate() {
      let i = k + 1;
      let next =
        prev + (self.theta.call(T::from_usize_(i) * dt) - self.alpha * prev) * dt + diff_scale * *z;
      *z = next;
      prev = next;
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
