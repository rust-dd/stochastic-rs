//! # Adg
//!
//! $$
//! dX_{i,t}=(k(t)-\theta(t)X_{i,t})dt+\sigma_i dW_{i,t},\quad
//! r_i(t)=\phi(t)+b(t)X_{i,t}+c(t)X_{i,t}^2
//! $$
//!
//! `xn` independent latent factors, each an affine/quadratic-Gaussian-style
//! diffusion with a time-dependent additive drift target `k(t)` and a
//! time-dependent mean-reversion speed `θ(t)`, observed through a
//! quadratic output map `phi(t) + b(t)X + c(t)X²`.
//!
use ndarray::Array1;
use ndarray::Array2;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

pub struct Adg<T: FloatExt, S: SeedExt = Unseeded> {
  /// Time-dependent additive drift target function k(t) — the level each
  /// latent factor is pulled toward, analogous to Hull-White's θ(t).
  pub k: Fn1D<T>,
  /// Time-dependent mean-reversion speed function θ(t), multiplying the
  /// pull-back term `−θ(t)·X_t`. Despite the name, this is a speed, not
  /// a level — see `k` for the level.
  pub theta: Fn1D<T>,
  /// Per-factor diffusion scale vector σ_i, one entry per latent factor
  /// (length `xn`).
  pub sigma: Array1<T>,
  /// Time-dependent intercept φ(t) of the output observation equation
  /// `r = φ(t) + b(t)·X + c(t)·X²`.
  pub phi: Fn1D<T>,
  /// Time-dependent linear loading b(t) on the latent factor in the
  /// observation equation.
  pub b: Fn1D<T>,
  /// Time-dependent quadratic loading c(t) on the squared latent factor
  /// in the observation equation (the "quadratic" in quadratic-Gaussian).
  pub c: Fn1D<T>,
  /// Number of time steps per latent-factor path.
  pub n: usize,
  /// Number of independent latent factors (rows of the output matrix);
  /// each evolves as its own `dX_i = (k(t) − θ(t)X_i)dt + σ_i dW_i`.
  pub xn: usize,
  /// Initial values `X_i(0)` for each latent factor, length `xn`.
  pub x0: Array1<T>,
  /// Simulation horizon [0, t] shared by all `xn` latent-factor paths
  /// (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Adg<T, S> {
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
    seed: S,
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
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Adg<T, S> {
  type Output = Array2<T>;
  type Sampler<'s>
    = AdgSampler<'s, T, S>
  where
    Self: 's;

  /// Derives a seed once, at construction, for [`AdgSampler`] to own.
  /// Deriving (not cloning) is what decorrelates chunks: the derived value
  /// is `self.seed`'s *mixed* next tick, not a raw snapshot, so chunk `i`'s
  /// basis and chunk `i+1`'s basis are hash-scrambled relative to each
  /// other rather than one raw stride apart. The drift, level and
  /// observation maps are user-supplied [`Fn1D`] callables (not clonable,
  /// since the Python variant holds a `pyo3::Py`) so there is nothing else
  /// reusable to hoist across calls beyond the borrowed process itself;
  /// the per-row `sample_inner`'s `SimdNormal::new(..., seed)` calls
  /// consume this owned seed directly, the same ticks the legacy code
  /// consumed from `self.seed` per call, so the first path reproduces the
  /// legacy stream bit-for-bit.
  fn sampler(&self) -> AdgSampler<'_, T, S> {
    AdgSampler {
      adg: self,
      seed: self.seed.derive(),
    }
  }
}

/// Reusable [`Adg`] sampler: borrows the process and owns a seed derived
/// once at construction. Each row's Gaussian increments are generated
/// inside the step body from that owned seed.
#[doc(hidden)]
pub struct AdgSampler<'a, T: FloatExt, S: SeedExt> {
  adg: &'a Adg<T, S>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for AdgSampler<'_, T, S> {
  type Output = Array2<T>;

  fn sample_into(&mut self, out: &mut Array2<T>) {
    *out = self.adg.sample_inner(&self.seed);
  }

  fn sample(&mut self) -> Array2<T> {
    self.adg.sample_inner(&self.seed)
  }
}

impl<T: FloatExt, S: SeedExt> Adg<T, S> {
  fn sample_inner(&self, seed: &S) -> Array2<T> {
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
      let normal = SimdNormal::<T>::new(T::zero(), sqrt_dt, seed);
      normal.fill_slice(tail);

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
        seeded: Some(Adg::new(
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
          Deterministic::new(s),
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
      Unseeded,
    );
    let path = adg.sample();
    assert_eq!(path.nrows(), xn);
    assert_eq!(path.ncols(), 32);
  }
}
