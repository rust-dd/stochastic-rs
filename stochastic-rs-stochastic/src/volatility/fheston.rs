//! # Fheston
//!
//! $$
//! dS_t=\mu S_tdt+\sqrt{v_t}S_tdW_t,\quad dv_t=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dB_t^H
//! $$
//!
//! References:
//! - Gatheral J., Jaisson T., Rosenbaum M. (2018) — *Volatility Is
//!   Rough*, Quantitative Finance 18(6), 933–949,
//!   DOI: 10.1080/14697688.2017.1393551.
//! - El Euch O. & Rosenbaum M. (2019) — *The Characteristic Function of
//!   Rough Heston Models*, Mathematical Finance 29(1), 3–38,
//!   DOI: 10.1111/mafi.12173 — defines the Volterra `v_t` above.
//!
//! `fill_paths` below approximates the exact fractional-kernel Volterra
//! convolution with a single OU-type carrier factor `zt` plus a direct
//! (`O(n²)`) memory sum, scaled by free `c1`/`c2` coefficients — an
//! approximate lifting scheme in the spirit of Abi Jaber & El Euch
//! (2019) — *Multifactor Approximation of Rough Volatility Models*,
//! SIAM Journal on Financial Mathematics 10(2), 309–349,
//! DOI: 10.1137/18M1170236 — but it is this crate's own one-factor
//! simplification, not a reproduction of that (or any other specific
//! published) numerical scheme.
//!
use ndarray::Array1;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::special::gamma;

use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

pub struct RoughHeston<T: FloatExt, S: SeedExt = Unseeded> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
  /// Initial variance/volatility level.
  pub v0: Option<T>,
  /// Long-run variance level θ the Markovian lifting factor `yt` reverts
  /// toward (θ in the module header).
  pub theta: T,
  /// Mean-reversion speed κ of the Markovian lifting factor `yt`.
  pub kappa: T,
  /// Vol-of-vol ξ scaling both the local (`zt`) and memory (Volterra
  /// integral) correction terms that lift the exact fractional kernel into
  /// a bounded-state approximation.
  pub nu: T,
  /// Calibration coefficient scaling the local correction term `zt` in the
  /// two-term rational approximation of the fractional kernel; `None`
  /// defaults to 1 (the untruncated coefficient).
  pub c1: Option<T>,
  /// Calibration coefficient scaling the memory (Volterra-integral)
  /// correction term in the same approximation; `None` defaults to 1.
  pub c2: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Number of points sampled along the rough-Heston path.
  pub n: usize,
  /// Drift of the log-price process (default 0).
  pub mu: Option<T>,
  /// Initial price level (default 1).
  pub s0: Option<T>,
  /// Correlation between price and vol innovations (default 0 = independent).
  pub rho: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> RoughHeston<T, S> {
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
    seed: S,
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
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for RoughHeston<T, S> {
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = RoughHestonSampler<T, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler, matching
  /// `Cgns`'s and `Bates1996`'s own `sampler()` shape. **Correction to an
  /// earlier verdict:** this was once documented as a full exception —
  /// "correlated-Gaussian source ignores `self.seed` entirely" — on the
  /// grounds that `cgns` was built fresh with the literal `Unseeded` and
  /// consumed via a bare `.sample()`. That reasoning was wrong: `Cgns::
  /// sample_impl<S2: SeedExt>(&self, seed: &S2)` is generic over an external
  /// seed, exactly like every sibling `cgns`-holding type
  /// (`DuffieKan`/`BatesSvj`/`DoubleHeston`/`Hkde`) already uses — the bare
  /// `.sample()` was a plain bug, not a structural limitation. Fixed the
  /// same way `Bates1996`'s identical bug was fixed. Unlike `Bates1996`,
  /// `RoughHeston` has no jump component, so this fix makes the type
  /// **fully** seed-reproducible, not merely partially — it carries no
  /// exception at all now.
  fn sampler(&self) -> RoughHestonSampler<T, S> {
    let n_steps = self.n.saturating_sub(1);
    let dt = if n_steps > 0 {
      self.t.unwrap_or(T::one()) / T::from_usize_(n_steps)
    } else {
      T::zero()
    };
    // `cgns`'s own seed stays the dead `Unseeded` default — the sampler
    // drives it via `sample_impl(&self.seed)` below instead, exactly like
    // `Bates1996`'s private `cgns` field.
    let rho = self.rho.unwrap_or(T::zero());
    RoughHestonSampler {
      n: self.n,
      hurst: self.hurst,
      theta: self.theta,
      kappa: self.kappa,
      nu: self.nu,
      c1: self.c1.unwrap_or(T::one()),
      c2: self.c2.unwrap_or(T::one()),
      mu: self.mu.unwrap_or(T::zero()),
      s0: self.s0.unwrap_or(T::one()),
      v0_sq: self.v0.unwrap_or(T::one()).powi(2),
      dt,
      g: gamma(self.hurst.to_f64().unwrap() - 0.5),
      cgns: Cgns::new(rho, n_steps, self.t, Unseeded),
      seed: self.seed.derive(),
    }
  }
}

/// Reusable [`RoughHeston`] sampling state: owns the correlated-Gaussian
/// generator, an owned already-derived seed to drive it, and the
/// precomputed Volterra-kernel constants, so a Monte-Carlo loop reuses both
/// output buffers.
#[doc(hidden)]
pub struct RoughHestonSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  hurst: T,
  theta: T,
  kappa: T,
  nu: T,
  c1: T,
  c2: T,
  mu: T,
  s0: T,
  v0_sq: T,
  dt: T,
  g: f64,
  cgns: Cgns<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> RoughHestonSampler<T, S> {
  fn fill_paths(&mut self, s: &mut [T], v2: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;

    let [gn_vol, gn_price] = self.cgns.sample_impl(&self.seed);

    let mut yt = Array1::<T>::zeros(self.n);
    let mut zt = Array1::<T>::zeros(self.n);
    let mut sigma_tilde2 = Array1::<T>::zeros(self.n);

    let v0_sq = self.v0_sq;
    let mu = self.mu;

    yt[0] = v0_sq;
    zt[0] = T::zero();
    sigma_tilde2[0] = v0_sq;
    v2[0] = v0_sq;
    s[0] = self.s0;
    let g = self.g;
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

      v2[i] =
        yt[i] + self.c1 * self.nu * zt[i] + self.c2 * self.nu * integral / T::from_f64_fast(g);

      // Price path: gn_price is already rho-correlated with gn_vol via Cgns
      let vi = v2[i - 1].max(T::zero());
      let log_inc = (mu - half * vi) * dt + vi.sqrt() * gn_price[i - 1];
      s[i] = s[i - 1] * log_inc.exp();
    }
  }
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for RoughHestonSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [s, v2] = out;
    self.fill_paths(
      s.as_slice_mut()
        .expect("RoughHeston output must be contiguous"),
      v2.as_slice_mut()
        .expect("RoughHeston output must be contiguous"),
    );
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut s = Array1::<T>::zeros(self.n);
    let mut v2 = Array1::<T>::zeros(self.n);
    self.fill_paths(
      s.as_slice_mut().expect("contiguous"),
      v2.as_slice_mut().expect("contiguous"),
    );
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
          Deterministic::new(sd),
        );
        m.mu = mu.map(|v| v as f32);
        m.s0 = s0.map(|v| v as f32);
        m.rho = rho.map(|v| v as f32);
        obj.seeded_f32 = Some(m);
      }
      (Some(sd), _) => {
        let mut m = RoughHeston::new(
          hurst,
          v0,
          theta,
          kappa,
          nu,
          c1,
          c2,
          t,
          n,
          Deterministic::new(sd),
        );
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
          Unseeded,
        );
        m.mu = mu.map(|v| v as f32);
        m.s0 = s0.map(|v| v as f32);
        m.rho = rho.map(|v| v as f32);
        obj.inner_f32 = Some(m);
      }
      (None, _) => {
        let mut m = RoughHeston::new(hurst, v0, theta, kappa, nu, c1, c2, t, n, Unseeded);
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
