//! # Duffie Kan
//!
//! $$
//! dX_t=K(\Theta-X_t)dt+\sqrt{A+BX_t}\,dW_t,\quad r_t=\ell_0+\ell^\top X_t
//! $$
//!
use ndarray::Array1;

use crate::stochastic::noise::cgns::CGNS;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Standard Duffie–Kan two-factor model (continuous, no jumps).
pub struct DuffieKan<T: FloatExt> {
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
  cgns: CGNS<T>,
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
      cgns: CGNS::new(rho, n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for DuffieKan<T> {
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
  sig: (alpha, beta, gamma_, rho, a1, b1, c1, sigma1, a2, b2, c2, sigma2, n, r0=None, x0=None, t=None, dtype=None),
  params: (alpha: f64, beta: f64, gamma_: f64, rho: f64, a1: f64, b1: f64, c1: f64, sigma1: f64, a2: f64, b2: f64, c2: f64, sigma2: f64, n: usize, r0: Option<f64>, x0: Option<f64>, t: Option<f64>)
);