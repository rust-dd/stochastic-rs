//! # Adg
//!
//! $$
//! dX_t=K(\Theta-X_t)dt+\sqrt{A+BX_t}\,dW_t,\quad r_t=\ell_0+\ell^\top X_t
//! $$
//!
use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::ProcessExt;

pub struct Adg<T: FloatExt, S: SeedExt = Unseeded> {
  /// Jump-size adjustment / shape parameter.
  pub k: Fn1D<T>,
  /// Long-run target level / model location parameter.
  pub theta: Fn1D<T>,
  /// Diffusion / noise scale parameter.
  pub sigma: Array1<T>,
  /// Autoregressive coefficient vector.
  pub phi: Fn1D<T>,
  /// Model coefficient / user-supplied diffusion term.
  pub b: Fn1D<T>,
  /// Model coefficient for nonlinear drift/level terms.
  pub c: Fn1D<T>,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Model parameter controlling process dynamics.
  pub xn: usize,
  /// Initial value of the primary state variable.
  pub x0: Array1<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt> Adg<T> {
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
    assert_eq!(
      sigma.len(),
      xn,
      "sigma length ({}) must match xn ({})",
      sigma.len(),
      xn
    );
    assert_eq!(
      x0.len(),
      xn,
      "x0 length ({}) must match xn ({})",
      x0.len(),
      xn
    );
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
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> Adg<T, Deterministic> {
  pub fn seeded(
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
    seed: u64,
  ) -> Self {
    assert_eq!(
      sigma.len(),
      xn,
      "sigma length ({}) must match xn ({})",
      sigma.len(),
      xn
    );
    assert_eq!(
      x0.len(),
      xn,
      "x0 length ({}) must match xn ({})",
      x0.len(),
      xn
    );
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
      seed: Deterministic::new(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Adg<T, S> {
  type Output = Array2<T>;

  fn sample(&self) -> Self::Output {
    let dt = if self.n > 1 {
      self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
    } else {
      T::zero()
    };
    let sqrt_dt = dt.sqrt();

    let mut adg = Array2::<T>::zeros((self.xn, self.n));
    for i in 0..self.xn {
      let mut row = adg.row_mut(i);
      let row_slice = row
        .as_slice_mut()
        .expect("Adg state row must be contiguous in memory");
      row_slice[0] = self.x0[i];
      if self.n <= 1 {
        continue;
      }

      let tail = &mut row_slice[1..];
      let normal = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &self.seed);
      normal.fill_slice_fast(tail);

      for j in 1..self.n {
        let t = T::from_usize_(j) * dt;
        row_slice[j] = row_slice[j - 1]
          + (self.k.call(t) - self.theta.call(t) * row_slice[j - 1]) * dt
          + self.sigma[i] * row_slice[j];
      }
    }

    let mut r = Array2::zeros((self.xn, self.n));

    for i in 0..self.xn {
      for j in 0..self.n {
        let t = T::from_usize_(j) * dt;
        let x = adg[(i, j)];
        r[(i, j)] = self.phi.call(t) + self.b.call(t) * x + self.c.call(t) * x * x;
      }
    }

    r
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyAdg {
  inner: Option<Adg<f64>>,
  seeded: Option<Adg<f64, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyAdg {
  #[new]
  #[pyo3(signature = (k, theta, sigma, phi, b, c, n, xn, x0, t=None, seed=None))]
  fn new(
    k: pyo3::Py<pyo3::PyAny>,
    theta: pyo3::Py<pyo3::PyAny>,
    sigma: Vec<f64>,
    phi: pyo3::Py<pyo3::PyAny>,
    b: pyo3::Py<pyo3::PyAny>,
    c: pyo3::Py<pyo3::PyAny>,
    n: usize,
    xn: usize,
    x0: Vec<f64>,
    t: Option<f64>,
    seed: Option<u64>,
  ) -> Self {
    match seed {
      Some(s) => Self {
        inner: None,
        seeded: Some(Adg::seeded(
          Fn1D::Py(k),
          Fn1D::Py(theta),
          ndarray::Array1::from_vec(sigma),
          Fn1D::Py(phi),
          Fn1D::Py(b),
          Fn1D::Py(c),
          n,
          xn,
          ndarray::Array1::from_vec(x0),
          t,
          s,
        )),
      },
      None => Self {
        inner: Some(Adg::new(
          Fn1D::Py(k),
          Fn1D::Py(theta),
          ndarray::Array1::from_vec(sigma),
          Fn1D::Py(phi),
          Fn1D::Py(b),
          Fn1D::Py(c),
          n,
          xn,
          ndarray::Array1::from_vec(x0),
          t,
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
  use ndarray::Array1;

  use super::*;

  fn const_one(_t: f64) -> f64 {
    1.0
  }
  fn const_zero(_t: f64) -> f64 {
    0.0
  }

  #[test]
  fn sample_runs() {
    let xn = 2;
    let sigma = Array1::<f64>::from_vec(vec![0.01, 0.01]);
    let x0 = Array1::<f64>::from_vec(vec![0.05, 0.05]);
    let adg = Adg::<f64>::new(
      const_one as fn(f64) -> f64,
      const_one as fn(f64) -> f64,
      sigma,
      const_zero as fn(f64) -> f64,
      const_one as fn(f64) -> f64,
      const_zero as fn(f64) -> f64,
      32,
      xn,
      x0,
      Some(1.0),
    );
    let path = adg.sample();
    assert_eq!(path.nrows(), xn);
    assert_eq!(path.ncols(), 32);
  }
}
