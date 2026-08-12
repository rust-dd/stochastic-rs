//! # Sabr
//!
//! $$
//! dF_t=\alpha_t F_t^\beta dW_t^1,\quad d\alpha_t=\nu\alpha_t dW_t^2,\ d\langle W^1,W^2\rangle_t=\rho dt
//! $$
//!
//! $\alpha_t$ is the stochastic-volatility state (initial value: field
//! `alpha0`); $\nu$ is the vol-of-vol (field `nu`); $\beta$ is the CEV
//! exponent (field `beta`); $\rho$ is the instantaneous correlation
//! (field `rho`).
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Clone)]
pub struct Sabr<T: FloatExt, S: SeedExt = Unseeded> {
  /// Vol-of-vol $\nu$: diffusion coefficient of the volatility state
  /// $\alpha_t$ (see module docs for the SDE).
  pub nu: T,
  /// CEV exponent β ∈ [0, 1] — elasticity of the forward's own volatility
  /// (β=1 is lognormal SABR, β=0 is normal SABR).
  pub beta: T,
  /// Instantaneous correlation ρ between the forward's and volatility's
  /// driving Brownian motions.
  pub rho: T,
  /// Number of points sampled along the SABR path.
  pub n: usize,
  /// Initial forward-rate level.
  pub f0: Option<T>,
  /// Initial volatility level $\alpha_0$.
  pub alpha0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  cgns: Cgns<T>,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Sabr::default().with_beta(0.5).with_rho(-0.6)`.
impl<T: FloatExt, S: SeedExt> Sabr<T, S> {
  pub fn new(
    nu: T,
    beta: T,
    rho: T,
    n: usize,
    f0: Option<T>,
    alpha0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(
      beta >= T::zero() && beta <= T::one(),
      "beta must be in [0, 1] for Sabr"
    );
    assert!(nu >= T::zero(), "nu must be non-negative");
    if let Some(alpha0) = alpha0 {
      assert!(alpha0 >= T::zero(), "alpha0 must be non-negative");
    }

    Self {
      nu,
      beta,
      rho,
      n,
      f0,
      alpha0,
      t,
      seed,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
    }
  }

  /// Replace `nu`, all else unchanged.
  pub fn with_nu(mut self, nu: T) -> Self {
    assert!(nu >= T::zero(), "nu must be non-negative");
    self.nu = nu;
    self
  }

  /// Replace `beta`, all else unchanged.
  pub fn with_beta(mut self, beta: T) -> Self {
    assert!(
      beta >= T::zero() && beta <= T::one(),
      "beta must be in [0, 1] for Sabr"
    );
    self.beta = beta;
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

  /// Replace `f0`, all else unchanged.
  pub fn with_f0(mut self, f0: Option<T>) -> Self {
    self.f0 = f0;
    self
  }

  /// Replace `alpha0`, all else unchanged.
  pub fn with_alpha0(mut self, alpha0: Option<T>) -> Self {
    if let Some(a) = alpha0 {
      assert!(a >= T::zero(), "alpha0 must be non-negative");
    }
    self.alpha0 = alpha0;
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

/// ν=0.4, β=0.7, ρ=-0.3, f₀=1, α₀=0.3 — matches the crate's Sabr
/// visualization-gallery fixture
/// (`stochastic-rs-viz/src/tests/categories/volatility_and_sheet.rs`, which
/// itself runs at n=96, not the n=252 below). t=1, n=252 — one trading year
/// of daily steps (this crate's `Default` convention, not itself drawn
/// from that fixture).
impl<T: FloatExt> Default for Sabr<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::from_f64_fast(0.4),
      T::from_f64_fast(0.7),
      T::from_f64_fast(-0.3),
      252,
      Some(T::one()),
      Some(T::from_f64_fast(0.3)),
      Some(T::one()),
      Unseeded,
    )
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Sabr<T, S> {
  /// `[F path, α path]`: index 0 is the forward `F`, index 1 is the
  /// stochastic-volatility state `α` (see module docs for the SDE).
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = SabrSampler<T, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart. `fill_paths`
  /// then uses this owned seed directly (no further derive) — exactly one
  /// derive from `self.seed` per chunk, matching what the legacy per-call
  /// `derive()` consumed, so the first path reproduces the legacy stream
  /// bit-for-bit.
  fn sampler(&self) -> SabrSampler<T, S> {
    SabrSampler {
      n: self.n,
      f0: self.f0.unwrap_or(T::zero()),
      alpha0: self.alpha0.unwrap_or(T::zero()).max(T::zero()),
      nu: self.nu,
      beta: self.beta,
      dt: self.cgns.dt(),
      cgns: self.cgns,
      seed: self.seed.derive(),
    }
  }
}

/// Reusable [`Sabr`] sampling state: owns the correlated-Gaussian generator
/// and the seed source so a Monte-Carlo loop reuses both output buffers.
#[doc(hidden)]
pub struct SabrSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  f0: T,
  alpha0: T,
  nu: T,
  beta: T,
  dt: T,
  cgns: Cgns<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> SabrSampler<T, S> {
  fn fill_paths(&mut self, f_: &mut [T], v: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let [cgn1, cgn2] = &self.cgns.sample_impl(&self.seed);

    f_[0] = self.f0;
    v[0] = self.alpha0;

    for i in 1..self.n {
      let f_prev = f_[i - 1].max(T::zero());
      let v_prev = v[i - 1].max(T::zero());
      f_[i] = f_[i - 1] + v_prev * f_prev.powf(self.beta) * cgn1[i - 1];
      // Exact step for dα = ν α dW preserves non-negativity.
      v[i] =
        v_prev * (self.nu * cgn2[i - 1] - T::from_f64_fast(0.5) * self.nu.powi(2) * self.dt).exp();
    }
  }
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for SabrSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [f_arr, v_arr] = out;
    let f_ = f_arr
      .as_slice_mut()
      .expect("Sabr output must be contiguous");
    let v = v_arr
      .as_slice_mut()
      .expect("Sabr output must be contiguous");
    self.fill_paths(f_, v);
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut f_ = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);
    self.fill_paths(
      f_.as_slice_mut().expect("contiguous"),
      v.as_slice_mut().expect("contiguous"),
    );
    [f_, v]
  }
}

impl<T: FloatExt, S: SeedExt> Sabr<T, S> {
  /// Calculate the Malliavin derivative of the Sabr model
  ///
  /// The Malliavin derivative of the volaility process in the Sabr model is given by:
  /// D_r \sigma_t = \nu \sigma_t 1_{[0, T]}(r)
  pub fn malliavin_of_vol(&self) -> [Array1<T>; 3] {
    let [f, v] = self.sample();

    let mut malliavin = Array1::<T>::zeros(self.n);

    for i in 0..self.n {
      malliavin[i] = self.nu * *v.last().unwrap();
    }

    [f, v, malliavin]
  }
}

// Python-visible parameter names stay `alpha`/`v0` (pre-existing public
// API surface); they forward positionally into `Sabr::new`'s renamed
// `nu`/`alpha0` parameters, so the Python signature is unaffected.
py_process_2x1d!(PySabr, Sabr,
  sig: (alpha, beta, rho, n, f0=None, v0=None, t=None, seed=None, dtype=None),
  params: (alpha: f64, beta: f64, rho: f64, n: usize, f0: Option<f64>, v0: Option<f64>, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::ProcessExt;

  #[test]
  fn volatility_stays_non_negative() {
    let p = Sabr::new(
      0.4_f64,
      0.5,
      -0.3,
      256,
      Some(1.0),
      Some(0.2),
      Some(1.0),
      Unseeded,
    );
    let [_f, v] = p.sample();
    assert!(v.iter().all(|x| *x >= 0.0));
  }

  /// Guards the field-vs-doc contradiction fixed in A1-b: the module doc's
  /// `dα = ν α dW₂` means `nu` is vol-of-vol and `alpha0` is the initial
  /// volatility state. Construction must compile with those names and the
  /// vol-of-vol must actually drive the volatility path's dispersion.
  #[test]
  fn sabr_nu_drives_volatility_dispersion() {
    use stochastic_rs_core::simd_rng::Deterministic;

    let make = |nu: f64| {
      Sabr::new(
        nu,
        0.5,
        -0.3,
        512,
        Some(1.0),
        Some(0.2),
        Some(1.0),
        Deterministic::new(7),
      )
    };

    let small = make(0.05);
    let large = make(0.9);
    // Field-name guard: `nu` is vol-of-vol, `alpha0` is the initial state.
    assert_eq!(small.nu, 0.05);
    assert_eq!(large.alpha0, Some(0.2));

    let [_f_small, v_small] = small.sample();
    let [_f_large, v_large] = large.sample();

    let variance = |x: &Array1<f64>| {
      let mean = x.iter().copied().sum::<f64>() / x.len() as f64;
      x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64
    };

    assert!(
      variance(&v_large) > variance(&v_small),
      "large-nu volatility path must disperse more than small-nu: var(small)={}, var(large)={}",
      variance(&v_small),
      variance(&v_large)
    );
  }
}
