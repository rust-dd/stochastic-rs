//! # HJM
//!
//! $$
//! df(t,T)=\alpha(t,T)dt+\sigma(t,T)\,dW_t
//! $$
//!
use ndarray::Array1;

use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::Fn2D;
use crate::traits::ProcessExt;

/// HJM-style Euler simulator.
///
/// This implementation treats `r`, `p`, and `f` as user-driven SDE components and
/// does not enforce the no-arbitrage HJM drift restriction between `alpha` and `sigma`.
pub struct HJM<T: FloatExt, S: SeedExt = Unseeded> {
  /// Model coefficient / user-supplied drift term.
  pub a: Fn1D<T>,
  /// Model coefficient / user-supplied diffusion term.
  pub b: Fn1D<T>,
  /// Order / lag count for the autoregressive component.
  pub p: Fn2D<T>,
  /// Model coefficient / moving-average term parameter.
  pub q: Fn2D<T>,
  /// Volatility / variance level or coefficient.
  pub v: Fn2D<T>,
  /// Model shape / loading parameter.
  pub alpha: Fn2D<T>,
  /// Diffusion / noise scale parameter.
  pub sigma: Fn2D<T>,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial short-rate / interest-rate level.
  pub r0: Option<T>,
  /// Initial bond-price / auxiliary level.
  pub p0: Option<T>,
  /// Initial forward-rate level.
  pub f0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
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
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> HJM<T, Deterministic> {
  pub fn seeded(
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
    seed: u64,
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
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for HJM<T, S> {
  type Output = [Array1<T>; 3];

  fn sample(&self) -> Self::Output {
    let mut r = Array1::<T>::zeros(self.n);
    let mut p = Array1::<T>::zeros(self.n);
    let mut f_ = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return [r, p, f_];
    }

    r[0] = self.r0.unwrap_or(T::zero());
    p[0] = self.p0.unwrap_or(T::zero());
    f_[0] = self.f0.unwrap_or(T::zero());
    if self.n == 1 {
      return [r, p, f_];
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let sqrt_dt = dt.sqrt();
    let mut seed = self.seed;
    {
      let r_slice = r
        .as_slice_mut()
        .expect("HJM short-rate path must be contiguous in memory");
      let r_tail = &mut r_slice[1..];
      let normal_r = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &mut seed);
      normal_r.fill_slice_fast(r_tail);
    }
    {
      let p_slice = p
        .as_slice_mut()
        .expect("HJM bond-price path must be contiguous in memory");
      let p_tail = &mut p_slice[1..];
      let normal_p = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &mut seed);
      normal_p.fill_slice_fast(p_tail);
    }
    {
      let f_slice = f_
        .as_slice_mut()
        .expect("HJM forward-rate path must be contiguous in memory");
      let f_tail = &mut f_slice[1..];
      let normal_f = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &mut seed);
      normal_f.fill_slice_fast(f_tail);
    }

    let t_max = self.t.unwrap_or(T::one());

    for i in 1..self.n {
      let t = T::from_usize_(i) * dt;

      r[i] = r[i - 1] + self.a.call(t) * dt + self.b.call(t) * r[i];
      p[i] = p[i - 1]
        + self.p.call(t, t_max) * (self.q.call(t, t_max) * dt + self.v.call(t, t_max) * p[i]);
      f_[i] = f_[i - 1] + self.alpha.call(t, t_max) * dt + self.sigma.call(t, t_max) * f_[i];
    }

    [r, p, f_]
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyHJM {
  inner: Option<HJM<f64>>,
  seeded: Option<HJM<f64, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyHJM {
  #[new]
  #[pyo3(signature = (a, b, p, q, v, alpha, sigma, n, r0=None, p0=None, f0=None, t=None, seed=None))]
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
    seed: Option<u64>,
  ) -> Self {
    use crate::traits::Fn2D;
    match seed {
      Some(s) => Self {
        inner: None,
        seeded: Some(HJM::seeded(
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
          s,
        )),
      },
      None => Self {
        inner: Some(HJM::new(
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
        )),
        seeded: None,
      },
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
    py_dispatch_f64!(self, |inner| {
      let [a, b, c] = inner.sample();
      (
        a.into_pyarray(py).into_py_any(py).unwrap(),
        b.into_pyarray(py).into_py_any(py).unwrap(),
        c.into_pyarray(py).into_py_any(py).unwrap(),
      )
    })
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
