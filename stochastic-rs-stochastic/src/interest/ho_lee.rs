//! # Ho Lee
//!
//! $$
//! dr_t=\theta(t)dt+\sigma dW_t
//! $$
//!
use ndarray::Array1;
use ndarray::s;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::ProcessExt;

#[allow(non_snake_case)]
pub struct HoLee<T: FloatExt, S: SeedExt = Unseeded> {
  /// Model parameter controlling process dynamics.
  pub f_T: Option<Fn1D<T>>,
  /// Long-run target level / model location parameter.
  pub theta: Option<T>,
  /// Diffusion / noise scale parameter.
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> HoLee<T, S> {
  pub fn new(
    f_T: Option<Fn1D<T>>,
    theta: Option<T>,
    sigma: T,
    n: usize,
    t: Option<T>,
    seed: S,
  ) -> Self {
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
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for HoLee<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut r = Array1::<T>::zeros(self.n);
    if self.n <= 1 {
      return r;
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let sqrt_dt = dt.sqrt();
    let diff_scale = self.sigma;
    let mut prev = r[0];
    let mut tail_view = r.slice_mut(s![1..]);
    let tail = tail_view
      .as_slice_mut()
      .expect("HoLee output tail must be contiguous");
    let normal = SimdNormal::<T>::new(T::zero(), sqrt_dt, &self.seed);
    normal.fill_slice_fast(tail);

    for (k, z) in tail.iter_mut().enumerate() {
      let i = k + 1;
      let t = T::from_usize_(i) * dt;
      let drift = if let Some(ref f) = self.f_T {
        let eps = dt.max(T::from_f64_fast(1e-8));
        let t_minus = (t - eps).max(T::zero());
        let t_plus = t + eps;
        let df_dt = (f.call(t_plus) - f.call(t_minus)) / (t_plus - t_minus);
        df_dt + self.sigma.powf(T::from_usize_(2)) * t
      } else {
        self.theta.unwrap()
      };

      let next = prev + drift * dt + diff_scale * *z;
      *z = next;
      prev = next;
    }

    r
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::ProcessExt;

  fn f_curve(t: f64) -> f64 {
    t * t
  }

  #[test]
  fn uses_forward_curve_derivative_when_provided() {
    let p = HoLee::new(
      Some(Fn1D::Native(f_curve as fn(f64) -> f64)),
      None,
      0.0_f64,
      3,
      Some(1.0),
      Unseeded,
    );
    let r = p.sample();
    assert!((r[1] - 0.5).abs() < 1e-12);
    assert!((r[2] - 1.5).abs() < 1e-12);
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyHoLee {
  inner: Option<HoLee<f64>>,
  seeded: Option<HoLee<f64, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyHoLee {
  #[new]
  #[pyo3(signature = (sigma, n, f_T=None, theta=None, t=None, seed=None))]
  fn new(
    sigma: f64,
    n: usize,
    f_T: Option<pyo3::Py<pyo3::PyAny>>,
    theta: Option<f64>,
    t: Option<f64>,
    seed: Option<u64>,
  ) -> Self {
    match seed {
      Some(s) => Self {
        inner: None,
        seeded: Some(HoLee::new(
          f_T.map(Fn1D::Py),
          theta,
          sigma,
          n,
          t,
          Deterministic::new(s),
        )),
      },
      None => Self {
        inner: Some(HoLee::new(f_T.map(Fn1D::Py), theta, sigma, n, t, Unseeded)),
        seeded: None,
      },
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch_f64!(self, |inner| inner
      .sample()
      .into_pyarray(py)
      .into_py_any(py)
      .unwrap())
  }
}
