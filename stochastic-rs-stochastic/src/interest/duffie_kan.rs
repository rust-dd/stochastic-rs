//! # Duffie Kan
//!
//! $$
//! dX_t=K(\Theta-X_t)dt+\sqrt{A+BX_t}\,dW_t,\quad r_t=\ell_0+\ell^\top X_t
//! $$
//!
use ndarray::Array1;

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use crate::noise::cgns::CGNS;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Standard Duffie–Kan two-factor model (continuous, no jumps).
pub struct DuffieKan<T: FloatExt, S: SeedExt = Unseeded> {
  /// Model shape / loading parameter.
  pub alpha: T,
  /// Model slope / loading parameter.
  pub beta: T,
  /// Model asymmetry / nonlinearity parameter.
  pub gamma: T,
  /// Instantaneous correlation parameter.
  pub rho: T,
  /// Model coefficient for factor 1.
  pub a1: T,
  /// Model coefficient for factor 1.
  pub b1: T,
  /// Model coefficient for factor 1.
  pub c1: T,
  /// Diffusion/noise scale for factor 1.
  pub sigma1: T,
  /// Model coefficient for factor 2.
  pub a2: T,
  /// Model coefficient for factor 2.
  pub b2: T,
  /// Model coefficient for factor 2.
  pub c2: T,
  /// Diffusion/noise scale for factor 2.
  pub sigma2: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial short-rate / interest-rate level.
  pub r0: Option<T>,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
  cgns: CGNS<T, S>,
}

impl<T: FloatExt> DuffieKan<T> {
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
  ) -> Self {
    Self {
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
      seed: Unseeded,
      cgns: CGNS::new(rho, n - 1, t),
    }
  }
}

impl<T: FloatExt> DuffieKan<T, Deterministic> {
  pub fn seeded(
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
    seed: u64,
  ) -> Self {
    let mut s = Deterministic(seed);
    let child = s.derive();
    Self {
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
      seed: Deterministic(seed),
      cgns: CGNS::seeded(rho, n - 1, t, child.0),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for DuffieKan<T, S> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let dt = self.cgns.dt();
    let [cgn1, cgn2] = &self.cgns.sample();

    let mut r = Array1::<T>::zeros(self.n);
    let mut x = Array1::<T>::zeros(self.n);

    r[0] = self.r0.unwrap_or(T::zero());
    x[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      r[i] = r[i - 1]
        + (self.a1 * r[i - 1] + self.b1 * x[i - 1] + self.c1) * dt
        + self.sigma1 * (self.alpha * r[i - 1] + self.beta * x[i - 1] + self.gamma) * cgn1[i - 1];
      x[i] = x[i - 1]
        + (self.a2 * r[i - 1] + self.b2 * x[i - 1] + self.c2) * dt
        + self.sigma2 * (self.alpha * r[i - 1] + self.beta * x[i - 1] + self.gamma) * cgn2[i - 1];
    }

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
      0.5, 0.04, 0.5, -0.3, 0.01, 0.0, 0.0, 0.01, 0.0, 0.5, 0.0, 0.005, 64,
      Some(0.05), Some(0.05), Some(1.0),
    );
    let [r, x] = dk.sample();
    assert_eq!(r.len(), 64);
    assert_eq!(x.len(), 64);
  }
}
