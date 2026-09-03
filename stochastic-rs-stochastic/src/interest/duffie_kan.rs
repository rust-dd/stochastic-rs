//! # Duffie Kan
//!
//! $$
//! dX_t=K(\Theta-X_t)dt+\sqrt{A+BX_t}\,dW_t,\quad r_t=\ell_0+\ell^\top X_t
//! $$
//!
use std::marker::PhantomData;

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::device::HostBackend;
use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Standard Duffie–Kan two-factor model (continuous, no jumps). The
/// per-factor drift and diffusion-loading coefficients below (`a1..c2`,
/// `alpha`/`beta`/`gamma`) are the concrete, fixed-numeric form of the
/// module header's abstract `K(Θ-X)dt+√(A+BX)dW` — see each field's own
/// doc for exactly which term it multiplies.
///
/// Every field has a matching `with_*` builder setter, e.g.
/// `DuffieKan::new(..).with_rho(-0.5).with_sigma1(0.02)`.
pub struct DuffieKan<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Diffusion-loading coefficient on `r_t`, shared between both factors'
  /// diffusion scaling (multiplies `r` inside `alpha*r + beta*x + gamma`,
  /// itself scaled again by `sigma1`/`sigma2`).
  pub alpha: T,
  /// Diffusion-loading coefficient on `x_t`, shared between both factors'
  /// diffusion scaling.
  pub beta: T,
  /// Diffusion-loading intercept (constant term), shared between both
  /// factors' diffusion scaling.
  pub gamma: T,
  /// Instantaneous correlation ρ between the two driving Brownian motions
  /// `dW1`/`dW2`.
  pub rho: T,
  /// Coefficient of `r_t` in r's own drift (factor-1 equation).
  pub a1: T,
  /// Coefficient of `x_t` in r's drift (cross-factor coupling,
  /// factor-1 equation).
  pub b1: T,
  /// Constant drift intercept for r's equation (factor-1 equation).
  pub c1: T,
  /// Diffusion scale multiplying r's shared affine loading
  /// (factor-1 equation).
  pub sigma1: T,
  /// Coefficient of `r_t` in x's drift (cross-factor coupling,
  /// factor-2 equation).
  pub a2: T,
  /// Coefficient of `x_t` in x's own drift (factor-2 equation).
  pub b2: T,
  /// Constant drift intercept for x's equation (factor-2 equation).
  pub c2: T,
  /// Diffusion scale multiplying x's shared affine loading
  /// (factor-2 equation).
  pub sigma2: T,
  /// Number of points sampled along each of the `r`/`x` paths.
  pub n: usize,
  /// Initial short rate r₀.
  pub r0: Option<T>,
  /// Initial value X₀ of the auxiliary factor.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] shared by both factors (defaults to 1 when
  /// omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  cgns: Cgns<T>,
  /// Sampling backend marker (compile-time): [`Cpu`] by default, a device
  /// marker after [`on`](Self::on). Public so `..Default::default()` struct
  /// updates keep working; it carries no data.
  pub backend: PhantomData<B>,
}

impl<T: FloatExt, S: SeedExt> DuffieKan<T, S> {
  pub fn new(
    alpha: T,
    beta: T,
    gamma: T,
    rho: T,
    a1: T,
    b1: T,
    c1: T,
    sigma1: T,
    a2: T,
    b2: T,
    c2: T,
    sigma2: T,
    n: usize,
    r0: Option<T>,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self {
      backend: PhantomData,
      alpha,
      beta,
      gamma,
      rho,
      a1,
      b1,
      c1,
      sigma1,
      a2,
      b2,
      c2,
      sigma2,
      n,
      r0,
      x0,
      t,
      seed,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> DuffieKan<T, S, B> {
  /// Replace `alpha`, all else unchanged.
  pub fn with_alpha(mut self, alpha: T) -> Self {
    self.alpha = alpha;
    self
  }

  /// Replace `beta`, all else unchanged.
  pub fn with_beta(mut self, beta: T) -> Self {
    self.beta = beta;
    self
  }

  /// Replace `gamma`, all else unchanged.
  pub fn with_gamma(mut self, gamma: T) -> Self {
    self.gamma = gamma;
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

  /// Replace `a1`, all else unchanged.
  pub fn with_a1(mut self, a1: T) -> Self {
    self.a1 = a1;
    self
  }

  /// Replace `b1`, all else unchanged.
  pub fn with_b1(mut self, b1: T) -> Self {
    self.b1 = b1;
    self
  }

  /// Replace `c1`, all else unchanged.
  pub fn with_c1(mut self, c1: T) -> Self {
    self.c1 = c1;
    self
  }

  /// Replace `sigma1`, all else unchanged.
  pub fn with_sigma1(mut self, sigma1: T) -> Self {
    self.sigma1 = sigma1;
    self
  }

  /// Replace `a2`, all else unchanged.
  pub fn with_a2(mut self, a2: T) -> Self {
    self.a2 = a2;
    self
  }

  /// Replace `b2`, all else unchanged.
  pub fn with_b2(mut self, b2: T) -> Self {
    self.b2 = b2;
    self
  }

  /// Replace `c2`, all else unchanged.
  pub fn with_c2(mut self, c2: T) -> Self {
    self.c2 = c2;
    self
  }

  /// Replace `sigma2`, all else unchanged.
  pub fn with_sigma2(mut self, sigma2: T) -> Self {
    self.sigma2 = sigma2;
    self
  }

  /// Replace `r0`, all else unchanged.
  pub fn with_r0(mut self, r0: Option<T>) -> Self {
    self.r0 = r0;
    self
  }

  /// Replace `x0`, all else unchanged.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    self.x0 = x0;
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

backend_switch!([T: FloatExt, S: SeedExt] DuffieKan<T, S> { alpha, beta, gamma, rho, a1, b1, c1, sigma1, a2, b2, c2, sigma2, n, r0, x0, t, seed, cgns } via host);

impl<T: FloatExt, S: SeedExt, B: HostBackend> ProcessExt<T> for DuffieKan<T, S, B> {
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = DuffieKanSampler<T, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart.
  fn sampler(&self) -> DuffieKanSampler<T, S> {
    DuffieKanSampler {
      n: self.n,
      r0: self.r0.unwrap_or(T::zero()),
      x0: self.x0.unwrap_or(T::zero()),
      alpha: self.alpha,
      beta: self.beta,
      gamma: self.gamma,
      a1: self.a1,
      b1: self.b1,
      c1: self.c1,
      sigma1: self.sigma1,
      a2: self.a2,
      b2: self.b2,
      c2: self.c2,
      sigma2: self.sigma2,
      dt: self.cgns.dt(),
      cgns: self.cgns,
      seed: self.seed.derive(),
    }
  }
}

/// Reusable [`DuffieKan`] sampling state: owns the correlated-Gaussian
/// generator and the seed source so a Monte-Carlo loop reuses both output
/// buffers.
#[doc(hidden)]
pub struct DuffieKanSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  r0: T,
  x0: T,
  alpha: T,
  beta: T,
  gamma: T,
  a1: T,
  b1: T,
  c1: T,
  sigma1: T,
  a2: T,
  b2: T,
  c2: T,
  sigma2: T,
  dt: T,
  cgns: Cgns<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> DuffieKanSampler<T, S> {
  fn fill_paths(&mut self, r: &mut [T], x: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;
    let [cgn1, cgn2] = &self.cgns.sample_impl(&self.seed);

    r[0] = self.r0;
    x[0] = self.x0;

    for i in 1..self.n {
      r[i] = r[i - 1]
        + (self.a1 * r[i - 1] + self.b1 * x[i - 1] + self.c1) * dt
        + self.sigma1 * (self.alpha * r[i - 1] + self.beta * x[i - 1] + self.gamma) * cgn1[i - 1];
      x[i] = x[i - 1]
        + (self.a2 * r[i - 1] + self.b2 * x[i - 1] + self.c2) * dt
        + self.sigma2 * (self.alpha * r[i - 1] + self.beta * x[i - 1] + self.gamma) * cgn2[i - 1];
    }
  }
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for DuffieKanSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [r_arr, x_arr] = out;
    let r = r_arr
      .as_slice_mut()
      .expect("DuffieKan output must be contiguous");
    let x = x_arr
      .as_slice_mut()
      .expect("DuffieKan output must be contiguous");
    self.fill_paths(r, x);
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut r = Array1::<T>::zeros(self.n);
    let mut x = Array1::<T>::zeros(self.n);
    self.fill_paths(
      r.as_slice_mut().expect("contiguous"),
      x.as_slice_mut().expect("contiguous"),
    );
    [r, x]
  }
}

py_process_2x1d!(PyDuffieKan, DuffieKan,
  sig: (alpha, beta, gamma_, rho, a1, b1, c1, sigma1, a2, b2, c2, sigma2, n, r0=None, x0=None, t=None, seed=None, dtype=None),
  params: (alpha: f64, beta: f64, gamma_: f64, rho: f64, a1: f64, b1: f64, c1: f64, sigma1: f64, a2: f64, b2: f64, c2: f64, sigma2: f64, n: usize, r0: Option<f64>, x0: Option<f64>, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn sample_returns_two_paths() {
    let dk = DuffieKan::<f64>::new(
      0.5,
      0.04,
      0.5,
      -0.3,
      0.01,
      0.0,
      0.0,
      0.01,
      0.0,
      0.5,
      0.0,
      0.005,
      64,
      Some(0.05),
      Some(0.05),
      Some(1.0),
      Unseeded,
    );
    let [r, x] = dk.sample();
    assert_eq!(r.len(), 64);
    assert_eq!(x.len(), 64);
  }
}
