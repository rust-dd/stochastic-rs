//! # Bergomi
//!
//! $$
//! dS_t=S_t\sqrt{v_t}\,dW_t^1,\quad v_t = v_0^2\,\exp\!\bigl(\nu\,W_t^2 - \tfrac12\nu^2 t\bigr)
//! $$
//!
//! **Scope (single-factor log-normal Bergomi, NOT the full Bergomi 2009
//! one-factor model).** This implementation evolves the spot variance as
//!
//! ```text
//! v(t_i) = v_0² · exp(ν · Σ_{j<i} cgn2_j  −  ½ ν² t_i)
//! ```
//!
//! where `cgn2_j` is a step of the correlated Gaussian noise process and
//! `Σ_{j<i} cgn2_j` is the discrete Brownian motion `W_{t_i}^2`. Compared
//! with the canonical Bergomi (2009) one-factor model
//! `v_t = ξ_0(t) exp(η X_t − ½ η² t^{2H})` with mean-reverting OU driver
//! `dX_t = -κ X_t dt + ν dW_t^2`, this implementation hard-codes:
//!
//! - **`H = ½`** (no roughness — the variance martingale correction is
//!   `½ η² t`, not `½ η² t^{2H}`).
//! - **`κ = 0`** (no mean-reversion of the variance driver — `X_t` reduces
//!   to a Brownian motion).
//! - **`ξ_0(t) ≡ v_0²`** (flat initial variance term-structure; no
//!   forward-variance curve input).
//!
//! Use this type for log-normal-vol smoke tests, GBM-with-stochastic-vol
//! sanity checks, or as a baseline for educational comparison. For a
//! genuine rough Bergomi (Volterra integral driver, `H < ½`) see
//! [`crate::volatility::rbergomi::RoughBergomi`] (which is itself a
//! scaled-Brownian approximation — see its module doc) or build a
//! high-fidelity simulator on top of [`crate::rough::MarkovLift`] or
//! [`crate::process::volterra::Volterra`].
//!
//! Reference: Bergomi, "Smile Dynamics II", Risk 18(10), 67-73 (2005);
//! Bergomi, "Stochastic Volatility Modeling" (2016) §7.
use ndarray::Array1;
use ndarray::s;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct Bergomi<T: FloatExt, S: SeedExt = Unseeded> {
  /// Volatility-of-volatility / tail-thickness parameter.
  pub nu: T,
  /// Initial variance/volatility level.
  pub v0: Option<T>,
  /// Initial asset/price level.
  pub s0: Option<T>,
  /// Risk-free rate / drift adjustment parameter.
  pub r: T,
  /// Instantaneous correlation parameter.
  pub rho: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
  cgns: Cgns<T>,
}

impl<T: FloatExt, S: SeedExt> Bergomi<T, S> {
  pub fn new(
    nu: T,
    v0: Option<T>,
    s0: Option<T>,
    r: T,
    rho: T,
    n: usize,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self {
      nu,
      v0,
      s0,
      r,
      rho,
      n,
      t,
      seed,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Bergomi<T, S> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let dt = self.cgns.dt();
    let [cgn1, cgn2] = &self.cgns.sample_impl(&self.seed.derive());

    let mut s = Array1::<T>::zeros(self.n);
    let mut v2 = Array1::<T>::zeros(self.n);
    s[0] = self.s0.unwrap_or(T::from_usize_(100));
    v2[0] = self.v0.unwrap_or(T::one()).powi(2);

    for i in 1..self.n {
      s[i] = s[i - 1] + self.r * s[i - 1] * dt + v2[i - 1].sqrt() * s[i - 1] * cgn1[i - 1];

      let sum_z = cgn2.slice(s![..i]).sum();
      let t = T::from_usize_(i) * dt;
      v2[i] = self.v0.unwrap_or(T::one()).powi(2)
        * (self.nu * sum_z - T::from_f64_fast(0.5) * self.nu.powi(2) * t).exp()
    }

    [s, v2]
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyBergomi {
  inner_f32: Option<Bergomi<f32>>,
  inner_f64: Option<Bergomi<f64>>,
  seeded_f32: Option<Bergomi<f32, crate::simd_rng::Deterministic>>,
  seeded_f64: Option<Bergomi<f64, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyBergomi {
  #[new]
  #[pyo3(signature = (nu, r, rho, n, v0=None, s0=None, t=None, seed=None, dtype=None))]
  fn new(
    nu: f64,
    r: f64,
    rho: f64,
    n: usize,
    v0: Option<f64>,
    s0: Option<f64>,
    t: Option<f64>,
    seed: Option<u64>,
    dtype: Option<&str>,
  ) -> Self {
    let mut s = Self {
      inner_f32: None,
      inner_f64: None,
      seeded_f32: None,
      seeded_f64: None,
    };
    match (seed, dtype.unwrap_or("f64")) {
      (Some(sd), "f32") => {
        s.seeded_f32 = Some(Bergomi::new(
          nu as f32,
          v0.map(|v| v as f32),
          s0.map(|v| v as f32),
          r as f32,
          rho as f32,
          n,
          t.map(|v| v as f32),
          Deterministic::new(sd),
        ));
      }
      (Some(sd), _) => {
        s.seeded_f64 = Some(Bergomi::new(
          nu,
          v0,
          s0,
          r,
          rho,
          n,
          t,
          Deterministic::new(sd),
        ));
      }
      (None, "f32") => {
        s.inner_f32 = Some(Bergomi::new(
          nu as f32,
          v0.map(|v| v as f32),
          s0.map(|v| v as f32),
          r as f32,
          rho as f32,
          n,
          t.map(|v| v as f32),
          Unseeded,
        ));
      }
      (None, _) => {
        s.inner_f64 = Some(Bergomi::new(nu, v0, s0, r, rho, n, t, Unseeded));
      }
    }
    s
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch!(self, |inner| {
      let [a, b] = inner.sample();
      (
        a.into_pyarray(py).into_py_any(py).unwrap(),
        b.into_pyarray(py).into_py_any(py).unwrap(),
      )
    })
  }

  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
    use numpy::IntoPyArray;
    use numpy::ndarray::Array2;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch!(self, |inner| {
      let samples = inner.sample_par(m);
      let n = samples[0][0].len();
      let mut r0 = Array2::zeros((m, n));
      let mut r1 = Array2::zeros((m, n));
      for (i, [a, b]) in samples.iter().enumerate() {
        r0.row_mut(i).assign(a);
        r1.row_mut(i).assign(b);
      }
      (
        r0.into_pyarray(py).into_py_any(py).unwrap(),
        r1.into_pyarray(py).into_py_any(py).unwrap(),
      )
    })
  }
}
