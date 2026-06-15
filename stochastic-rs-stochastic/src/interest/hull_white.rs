//! # Hull White
//!
//! $$
//! dr_t=\left(\theta(t)-a r_t\right)dt+\sigma dW_t
//! $$
//!
use ndarray::Array1;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

pub struct HullWhite<T: FloatExt, S: SeedExt = Unseeded> {
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
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> HullWhite<T, S> {
  pub fn new(
    theta: impl Into<Fn1D<T>>,
    alpha: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self {
      theta: theta.into(),
      alpha,
      sigma,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for HullWhite<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = HullWhiteSampler<'s, T>
  where
    Self: 's;

  fn sampler(&self) -> HullWhiteSampler<'_, T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    HullWhiteSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      alpha: self.alpha,
      diff_scale: self.sigma,
      theta: &self.theta,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`HullWhite`] sampling state. Borrows the process for its
/// time-dependent drift `θ(t)` and owns the Gaussian source so a Monte-Carlo
/// loop pays the `SimdNormal` setup once.
#[doc(hidden)]
pub struct HullWhiteSampler<'a, T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  alpha: T,
  diff_scale: T,
  theta: &'a Fn1D<T>,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> HullWhiteSampler<'_, T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let dt = self.dt;
    let diff_scale = self.diff_scale;
    let mut prev = out[0];
    let tail = &mut out[1..];
    self.normal.fill_slice_fast(tail);

    for (k, z) in tail.iter_mut().enumerate() {
      let i = k + 1;
      let next =
        prev + (self.theta.call(T::from_usize_(i) * dt) - self.alpha * prev) * dt + diff_scale * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for HullWhiteSampler<'_, T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("HullWhite output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyHullWhite {
  inner: Option<HullWhite<f64>>,
  seeded: Option<HullWhite<f64, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyHullWhite {
  #[new]
  #[pyo3(signature = (theta, alpha, sigma, n, x0=None, t=None, seed=None))]
  fn new(
    theta: pyo3::Py<pyo3::PyAny>,
    alpha: f64,
    sigma: f64,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    seed: Option<u64>,
  ) -> Self {
    match seed {
      Some(s) => Self {
        inner: None,
        seeded: Some(HullWhite::new(
          Fn1D::Py(theta),
          alpha,
          sigma,
          n,
          x0,
          t,
          Deterministic::new(s),
        )),
      },
      None => Self {
        inner: Some(HullWhite::new(
          Fn1D::Py(theta),
          alpha,
          sigma,
          n,
          x0,
          t,
          Unseeded,
        )),
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

#[cfg(test)]
mod tests {
  use super::*;

  fn const_theta(_t: f64) -> f64 {
    0.04
  }

  #[test]
  fn sample_length_matches_n() {
    let hw = HullWhite::<f64>::new(
      const_theta as fn(f64) -> f64,
      0.5,
      0.01,
      64,
      Some(0.04),
      Some(1.0),
      Unseeded,
    );
    let path = hw.sample();
    assert_eq!(path.len(), 64);
  }
}
