//! # Fheston
//!
//! $$
//! dS_t=\mu S_tdt+\sqrt{v_t}S_tdW_t,\quad dv_t=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dB_t^H
//! $$
//!
use ndarray::Array1;
use statrs::function::gamma::gamma;

use crate::distributions::normal::SimdNormal;
use crate::simd_rng::Deterministic;
use crate::simd_rng::SeedExt;
use crate::simd_rng::Unseeded;
use crate::stochastic::noise::cgns::CGNS;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct RoughHeston<T: FloatExt, S: SeedExt = Unseeded> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
  /// Initial variance/volatility level.
  pub v0: Option<T>,
  /// Long-run target level / model location parameter.
  pub theta: T,
  /// Mean-reversion speed parameter.
  pub kappa: T,
  /// Volatility-of-volatility / tail-thickness parameter.
  pub nu: T,
  /// Model coefficient for factor 1.
  pub c1: Option<T>,
  /// Model coefficient for factor 2.
  pub c2: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Drift of the log-price process (default 0).
  pub mu: Option<T>,
  /// Initial price level (default 1).
  pub s0: Option<T>,
  /// Correlation between price and vol innovations (default 0 = independent).
  pub rho: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt> RoughHeston<T> {
  pub fn new(
    hurst: T,
    v0: Option<T>,
    theta: T,
    kappa: T,
    nu: T,
    c1: Option<T>,
    c2: Option<T>,
    t: Option<T>,
    n: usize,
  ) -> Self {
    RoughHeston {
      hurst,
      v0,
      theta,
      kappa,
      nu,
      c1,
      c2,
      t,
      n,
      mu: None,
      s0: None,
      rho: None,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> RoughHeston<T, Deterministic> {
  pub fn seeded(
    hurst: T,
    v0: Option<T>,
    theta: T,
    kappa: T,
    nu: T,
    c1: Option<T>,
    c2: Option<T>,
    t: Option<T>,
    n: usize,
    seed: u64,
  ) -> Self {
    RoughHeston {
      hurst,
      v0,
      theta,
      kappa,
      nu,
      c1,
      c2,
      t,
      n,
      mu: None,
      s0: None,
      rho: None,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for RoughHeston<T, S> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let n_steps = self.n.saturating_sub(1);
    let dt = if n_steps > 0 {
      self.t.unwrap_or(T::one()) / T::from_usize_(n_steps)
    } else {
      T::zero()
    };

    // Use CGNS for rho-correlated noise: [gn_vol, gn_price]
    let rho = self.rho.unwrap_or(T::zero());
    let cgns = CGNS::new(rho, n_steps, self.t);
    let [gn_vol, gn_price] = cgns.sample();

    let mut yt = Array1::<T>::zeros(self.n);
    let mut zt = Array1::<T>::zeros(self.n);
    let mut sigma_tilde2 = Array1::<T>::zeros(self.n);
    let mut v2 = Array1::zeros(self.n);
    let mut s = Array1::zeros(self.n);

    if self.n == 0 {
      return [s, v2];
    }

    let v0_sq = self.v0.unwrap_or(T::one()).powi(2);
    let mu = self.mu.unwrap_or(T::zero());
    let s0 = self.s0.unwrap_or(T::one());

    yt[0] = v0_sq;
    zt[0] = T::zero();
    sigma_tilde2[0] = v0_sq;
    v2[0] = v0_sq;
    s[0] = s0;
    let g = gamma(self.hurst.to_f64().unwrap() - 0.5);
    let half = T::from_f64_fast(0.5);

    for i in 1..self.n {
      let t_i = dt * T::from_usize_(i);
      yt[i] = self.theta + (yt[i - 1] - self.theta) * (-self.kappa * dt).exp();
      zt[i] = zt[i - 1] * (-self.kappa * dt).exp()
        + sigma_tilde2[i - 1].max(T::zero()).sqrt() * gn_vol[i - 1];

      sigma_tilde2[i] = yt[i] + self.nu * zt[i];

      let integral = (0..i)
        .map(|j| {
          let tj = T::from_usize_(j) * dt;
          ((t_i - tj).powf(self.hurst - half) * zt[j]) * dt
        })
        .sum::<T>();

      v2[i] = yt[i]
        + self.c1.unwrap_or(T::one()) * self.nu * zt[i]
        + self.c2.unwrap_or(T::one()) * self.nu * integral / T::from_f64_fast(g);

      // Price path: gn_price is already rho-correlated with gn_vol via CGNS
      let vi = v2[i - 1].max(T::zero());
      let log_inc = (mu - half * vi) * dt + vi.sqrt() * gn_price[i - 1];
      s[i] = s[i - 1] * log_inc.exp();
    }

    [s, v2]
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyRoughHeston {
  inner_f32: Option<RoughHeston<f32>>,
  inner_f64: Option<RoughHeston<f64>>,
  seeded_f32: Option<RoughHeston<f32, crate::simd_rng::Deterministic>>,
  seeded_f64: Option<RoughHeston<f64, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyRoughHeston {
  #[new]
  #[pyo3(signature = (hurst, theta, kappa, nu, n, v0=None, c1=None, c2=None, t=None, mu=None, s0=None, rho=None, seed=None, dtype=None))]
  fn new(
    hurst: f64,
    theta: f64,
    kappa: f64,
    nu: f64,
    n: usize,
    v0: Option<f64>,
    c1: Option<f64>,
    c2: Option<f64>,
    t: Option<f64>,
    mu: Option<f64>,
    s0: Option<f64>,
    rho: Option<f64>,
    seed: Option<u64>,
    dtype: Option<&str>,
  ) -> Self {
    let mut obj = Self {
      inner_f32: None,
      inner_f64: None,
      seeded_f32: None,
      seeded_f64: None,
    };
    match (seed, dtype.unwrap_or("f64")) {
      (Some(sd), "f32") => {
        let mut m = RoughHeston::seeded(
          hurst as f32,
          v0.map(|v| v as f32),
          theta as f32,
          kappa as f32,
          nu as f32,
          c1.map(|v| v as f32),
          c2.map(|v| v as f32),
          t.map(|v| v as f32),
          n,
          sd,
        );
        m.mu = mu.map(|v| v as f32);
        m.s0 = s0.map(|v| v as f32);
        m.rho = rho.map(|v| v as f32);
        obj.seeded_f32 = Some(m);
      }
      (Some(sd), _) => {
        let mut m = RoughHeston::seeded(hurst, v0, theta, kappa, nu, c1, c2, t, n, sd);
        m.mu = mu;
        m.s0 = s0;
        m.rho = rho;
        obj.seeded_f64 = Some(m);
      }
      (None, "f32") => {
        let mut m = RoughHeston::new(
          hurst as f32,
          v0.map(|v| v as f32),
          theta as f32,
          kappa as f32,
          nu as f32,
          c1.map(|v| v as f32),
          c2.map(|v| v as f32),
          t.map(|v| v as f32),
          n,
        );
        m.mu = mu.map(|v| v as f32);
        m.s0 = s0.map(|v| v as f32);
        m.rho = rho.map(|v| v as f32);
        obj.inner_f32 = Some(m);
      }
      (None, _) => {
        let mut m = RoughHeston::new(hurst, v0, theta, kappa, nu, c1, c2, t, n);
        m.mu = mu;
        m.s0 = s0;
        m.rho = rho;
        obj.inner_f64 = Some(m);
      }
    }
    obj
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch!(self, |inner| {
      let [s, v] = inner.sample();
      (s.into_pyarray(py), v.into_pyarray(py))
        .into_py_any(py)
        .unwrap()
    })
  }

  fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use numpy::ndarray::Array2;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch!(self, |inner| {
      let paths = inner.sample_par(m);
      let n = paths[0][0].len();
      let mut s_result = Array2::zeros((m, n));
      let mut v_result = Array2::zeros((m, n));
      for (i, [s, v]) in paths.iter().enumerate() {
        s_result.row_mut(i).assign(s);
        v_result.row_mut(i).assign(v);
      }
      (s_result.into_pyarray(py), v_result.into_pyarray(py))
        .into_py_any(py)
        .unwrap()
    })
  }
}
