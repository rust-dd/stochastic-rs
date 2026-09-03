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
//! [`crate::volatility::rbergomi::RoughBergomi`] (hybrid-scheme Volterra
//! simulation — see its module doc) or build a simulator on top of
//! [`crate::rough::MarkovLift`] or [`crate::process::volterra::Volterra`].
//!
//! Reference: Bergomi, "Smile Dynamics II", Risk 18(10), 67-73 (2005);
//! Bergomi, "Stochastic Volatility Modeling" (2016) §7.
use std::marker::PhantomData;

use ndarray::Array1;
use ndarray::s;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::device::HostBackend;
use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Clone)]
pub struct Bergomi<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Vol-of-vol ν scaling the log-variance driver's dispersion.
  pub nu: T,
  /// Initial variance level v₀ (variance, not volatility — squared inside
  /// the sampler to seed `v(t_0) = v_0²`).
  pub v0: Option<T>,
  /// Initial asset price S₀.
  pub s0: Option<T>,
  /// Constant proportional drift rate of the asset (a GBM-style drift, not
  /// a correlation — that role belongs to `rho`).
  pub r: T,
  /// Correlation ρ between the asset's Brownian shock and the
  /// log-variance driver's innovations.
  pub rho: T,
  /// Number of points sampled along the Bergomi path.
  pub n: usize,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  cgns: Cgns<T>,
  /// Sampling backend marker (compile-time): [`Cpu`] by default, a device
  /// marker after [`on`](Self::on). Public so `..Default::default()` struct
  /// updates keep working; it carries no data.
  pub backend: PhantomData<B>,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Bergomi::default().with_nu(0.6).with_rho(-0.2)`.
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
      backend: PhantomData,
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

impl<T: FloatExt, S: SeedExt, B> Bergomi<T, S, B> {
  /// Replace `nu`, all else unchanged.
  pub fn with_nu(mut self, nu: T) -> Self {
    self.nu = nu;
    self
  }

  /// Replace `v0`, all else unchanged.
  pub fn with_v0(mut self, v0: Option<T>) -> Self {
    self.v0 = v0;
    self
  }

  /// Replace `s0`, all else unchanged.
  pub fn with_s0(mut self, s0: Option<T>) -> Self {
    self.s0 = s0;
    self
  }

  /// Replace `r`, all else unchanged.
  pub fn with_r(mut self, r: T) -> Self {
    self.r = r;
    self
  }

  /// Replace `rho`; rebuilds the cached correlated-Gaussian generator
  /// (`cgns`) so the new correlation actually reaches the sampler instead
  /// of a stale one computed from the old `rho`.
  pub fn with_rho(mut self, rho: T) -> Self {
    self.rho = rho;
    self.cgns = Cgns::new(rho, self.n - 1, self.t, Unseeded);
    self
  }

  /// Replace the number of simulation steps `n`; rebuilds the cached
  /// correlated-Gaussian generator, whose length and step size derive
  /// from `n`.
  pub fn with_steps(mut self, n: usize) -> Self {
    self.n = n;
    self.cgns = Cgns::new(self.rho, n - 1, self.t, Unseeded);
    self
  }

  /// Replace the simulation horizon `t`; rebuilds the cached
  /// correlated-Gaussian generator's step size, which derives from `t`.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    self.t = t;
    self.cgns = Cgns::new(self.rho, self.n - 1, t, Unseeded);
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// ν=0.4, v₀=0.2, s₀=100, r=0.01, ρ=-0.6 — a textbook Bergomi
/// parameterization. t=1, n=252 — one trading year of daily steps (this
/// crate's `Default` convention).
impl<T: FloatExt> Default for Bergomi<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::from_f64_fast(0.4),
      Some(T::from_f64_fast(0.2)),
      Some(T::from_f64_fast(100.0)),
      T::from_f64_fast(0.01),
      T::from_f64_fast(-0.6),
      252,
      Some(T::one()),
      Unseeded,
    )
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Bergomi<T, S> { nu, v0, s0, r, rho, n, t, seed, cgns } via host);

impl<T: FloatExt, S: SeedExt, B: HostBackend> ProcessExt<T> for Bergomi<T, S, B> {
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = BergomiSampler<T, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart.
  fn sampler(&self) -> BergomiSampler<T, S> {
    BergomiSampler {
      n: self.n,
      nu: self.nu,
      v0_sq: self.v0.unwrap_or(T::one()).powi(2),
      s0: self.s0.unwrap_or(T::from_usize_(100)),
      r: self.r,
      dt: self.cgns.dt(),
      cgns: self.cgns,
      seed: self.seed.derive(),
    }
  }
}

/// Reusable [`Bergomi`] sampling state: owns the correlated-Gaussian generator
/// and the seed source so a Monte-Carlo loop reuses both output buffers and the
/// noise setup.
#[doc(hidden)]
pub struct BergomiSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  nu: T,
  v0_sq: T,
  s0: T,
  r: T,
  dt: T,
  cgns: Cgns<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> BergomiSampler<T, S> {
  fn fill_paths(&mut self, s: &mut [T], v2: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;
    let [cgn1, cgn2] = &self.cgns.sample_impl(&self.seed);

    s[0] = self.s0;
    v2[0] = self.v0_sq;

    for i in 1..self.n {
      s[i] = s[i - 1] + self.r * s[i - 1] * dt + v2[i - 1].sqrt() * s[i - 1] * cgn1[i - 1];

      let sum_z = cgn2.slice(s![..i]).sum();
      let t = T::from_usize_(i) * dt;
      v2[i] = self.v0_sq * (self.nu * sum_z - T::from_f64_fast(0.5) * self.nu.powi(2) * t).exp()
    }
  }
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for BergomiSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [s, v2] = out;
    self.fill_paths(
      s.as_slice_mut().expect("Bergomi output must be contiguous"),
      v2.as_slice_mut()
        .expect("Bergomi output must be contiguous"),
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
