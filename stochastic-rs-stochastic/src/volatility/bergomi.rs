//! # Bergomi
//!
//! $$
//! dS_t=S_t\sqrt{v_t}dW_t^1,\quad v_t=\xi_0(t)\exp\!\left(\eta X_t-\tfrac12\eta^2 t^{2H}\right)
//! $$
//!
use ndarray::Array1;
use ndarray::s;

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use crate::noise::cgns::CGNS;
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
  cgns: CGNS<T>,
}

impl<T: FloatExt> Bergomi<T> {
  pub fn new(nu: T, v0: Option<T>, s0: Option<T>, r: T, rho: T, n: usize, t: Option<T>) -> Self {
    Self {
      nu,
      v0,
      s0,
      r,
      rho,
      n,
      t,
      seed: Unseeded,
      cgns: CGNS::new(rho, n - 1, t),
    }
  }
}

impl<T: FloatExt> Bergomi<T, Deterministic> {
  pub fn seeded(
    nu: T,
    v0: Option<T>,
    s0: Option<T>,
    r: T,
    rho: T,
    n: usize,
    t: Option<T>,
    seed: u64,
  ) -> Self {
    Self {
      nu,
      v0,
      s0,
      r,
      rho,
      n,
      t,
      seed: Deterministic(seed),
      cgns: CGNS::new(rho, n - 1, t),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Bergomi<T, S> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let dt = self.cgns.dt();
    let mut seed = self.seed;
    let [cgn1, cgn2] = &self.cgns.sample_impl(seed.derive());

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
        s.seeded_f32 = Some(Bergomi::seeded(
          nu as f32,
          v0.map(|v| v as f32),
          s0.map(|v| v as f32),
          r as f32,
          rho as f32,
          n,
          t.map(|v| v as f32),
          sd,
        ));
      }
      (Some(sd), _) => {
        s.seeded_f64 = Some(Bergomi::seeded(nu, v0, s0, r, rho, n, t, sd));
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
        ));
      }
      (None, _) => {
        s.inner_f64 = Some(Bergomi::new(nu, v0, s0, r, rho, n, t));
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
