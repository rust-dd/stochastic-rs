//! # Rbergomi
//!
//! $$
//! dS_t=S_t\sqrt{v_t}\,dW_t^1,\qquad
//! v(t_i) = v_0^2\,\exp\!\Bigl(\nu\sqrt{2H}\,t_i^{H-1/2}\sum_{j<i}\!Z_j
//!                         \;-\;\tfrac12\nu^2 t_i^{2H}\Bigr)
//! $$
//!
//! **Scope (scaled-Brownian-motion approximation of rough Bergomi —
//! variance-matched, NOT a true Volterra integral).** The canonical
//! Bayer-Friz-Gatheral (2016) rough Bergomi defines the log-variance
//! driver as the **Volterra fractional Brownian motion**
//!
//! ```text
//! W^H_t = √(2H) · ∫_0^t (t − s)^{H − ½} dW_s^2
//! ```
//!
//! whose autocovariance encodes long memory and pathwise roughness. This
//! implementation **does not** evaluate that Volterra integral. Instead
//! it uses the simpler discrete recipe
//!
//! ```text
//! v(t_i) = v_0² · exp[ ν · √(2H) · t_i^{H − ½} · Σ_{j<i} Z_j
//!                      − ½ ν² · t_i^{2H} ]
//! ```
//!
//! where the time-dependent kernel weight `(t_i − s_j)^{H − ½}` inside
//! the integrand has been **factored out** as a single multiplicative
//! factor `t_i^{H − ½}` applied to the cumulative sum of i.i.d. Gaussian
//! increments. The two coincide only when `H = ½` (Brownian case);
//! everywhere else this is an approximation that:
//!
//! - **preserves** the marginal variance scaling `Var[X_t] ∝ t^{2H}`
//!   (hence the `½ ν² t^{2H}` correction term matches the proper
//!   model);
//! - **does not preserve** the fBM autocovariance structure
//!   `Cov[X_s, X_t] ≠ ½(t^{2H} + s^{2H} − |t − s|^{2H})` — paths sampled
//!   here lack the long-memory / antipersistence kernel of true fBM.
//!
//! For applications where only the marginal variance and one-step
//! variance dynamics matter (e.g. teaching, smile-shape playgrounds,
//! Monte-Carlo stress tests of the BFG-style log-normal envelope), this
//! is fine. **For calibration / pricing where joint distributional
//! accuracy of `v_t` matters (caplets, swaptions, path-dependent rough
//! products), use a true Volterra simulator** — e.g.
//! [`crate::rough::MarkovLift`] (Bilokon-Wong 2026 generalised
//! Gauss-Laguerre exponential-sum representation; SIMD-batched) or
//! [`crate::rough::rl_heston::RlHeston`] / [`crate::rough::rl_bs::RlBs`]
//! built on top of it, or [`crate::process::volterra::Volterra`] for the
//! raw fractional integral.
//!
//! Reference: Bayer, Friz, Gatheral, "Pricing under rough volatility",
//! Quantitative Finance 16(6), 887-904 (2016) — for the canonical model
//! that this implementation approximates.
use ndarray::Array1;
use ndarray::s;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct RoughBergomi<T: FloatExt, S: SeedExt = Unseeded> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
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

impl<T: FloatExt> RoughBergomi<T> {
  pub fn new(
    hurst: T,
    nu: T,
    v0: Option<T>,
    s0: Option<T>,
    r: T,
    rho: T,
    n: usize,
    t: Option<T>,
  ) -> Self {
    RoughBergomi {
      hurst,
      nu,
      v0,
      s0,
      r,
      rho,
      n,
      t,
      seed: Unseeded,
      cgns: Cgns::new(rho, n - 1, t),
    }
  }
}

impl<T: FloatExt> RoughBergomi<T, Deterministic> {
  pub fn seeded(
    hurst: T,
    nu: T,
    v0: Option<T>,
    s0: Option<T>,
    r: T,
    rho: T,
    n: usize,
    t: Option<T>,
    seed: u64,
  ) -> Self {
    RoughBergomi {
      hurst,
      nu,
      v0,
      s0,
      r,
      rho,
      n,
      t,
      seed: Deterministic::new(seed),
      cgns: Cgns::new(rho, n - 1, t),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for RoughBergomi<T, S> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let dt = self.cgns.dt();
    let [cgn1, z] = &self.cgns.sample_impl(&self.seed.derive());

    let mut s = Array1::<T>::zeros(self.n);
    let mut v2 = Array1::<T>::zeros(self.n);
    s[0] = self.s0.unwrap_or(T::from_usize_(100));
    v2[0] = self.v0.unwrap_or(T::one()).powi(2);

    for i in 1..self.n {
      s[i] = s[i - 1] + self.r * s[i - 1] * dt + v2[i - 1].sqrt() * s[i - 1] * cgn1[i - 1];

      let sum_z = z.slice(s![..i]).sum();
      let t = T::from_usize_(i) * dt;
      v2[i] = self.v0.unwrap_or(T::one()).powi(2)
        * (self.nu
          * (T::from_usize_(2) * self.hurst).sqrt()
          * t.powf(self.hurst - T::from_f64_fast(0.5))
          * sum_z
          - T::from_f64_fast(0.5) * self.nu.powi(2) * t.powf(T::from_usize_(2) * self.hurst))
        .exp();
    }

    [s, v2]
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyRoughBergomi {
  inner_f32: Option<RoughBergomi<f32>>,
  inner_f64: Option<RoughBergomi<f64>>,
  seeded_f32: Option<RoughBergomi<f32, crate::simd_rng::Deterministic>>,
  seeded_f64: Option<RoughBergomi<f64, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyRoughBergomi {
  #[new]
  #[pyo3(signature = (hurst, nu, r, rho, n, v0=None, s0=None, t=None, seed=None, dtype=None))]
  fn new(
    hurst: f64,
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
        s.seeded_f32 = Some(RoughBergomi::seeded(
          hurst as f32,
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
        s.seeded_f64 = Some(RoughBergomi::seeded(hurst, nu, v0, s0, r, rho, n, t, sd));
      }
      (None, "f32") => {
        s.inner_f32 = Some(RoughBergomi::new(
          hurst as f32,
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
        s.inner_f64 = Some(RoughBergomi::new(hurst, nu, v0, s0, r, rho, n, t));
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
