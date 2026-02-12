//! A standard Duffie–Kan two-factor model without jumps.
//!
//! # Model Definition
//!
//! We consider two state variables, `r(t)` and `x(t)`, evolving under:
//!
//! \
//! \[
//!   dr(t) = \bigl(a_1 r(t) + b_1 x(t) + c_1\bigr)\,dt
//!            + \sigma_1 \bigl(\alpha\,r(t) + \beta\,x(t) + \gamma\bigr)\,dW_1(t),
//! \]
//! \[
//!   dx(t) = \bigl(a_2 r(t) + b_2 x(t) + c_2\bigr)\,dt
//!            + \sigma_2 \bigl(\alpha\,r(t) + \beta\,x(t) + \gamma\bigr)\,dW_2(t),
//! \]
//!
//! # Parameters
//! - `alpha, beta, gamma`: Linear combination for the diffusion part.
//! - `a1, b1, c1, sigma1`: Drift and diffusion coefficients for `r(t)`.
//! - `a2, b2, c2, sigma2`: Drift and diffusion coefficients for `x(t)`.
//! - `rho`: Correlation parameter is handled by the `cgns` correlated noise system
//!   (not used directly here, but embedded in the `CGNS` object).
//! - `r0, x0`: Initial values for `r(t)` and `x(t)`.
//! - `t`: Total time horizon.
//! - `n`: Number of time steps for simulation.
//! - `m`: Batch size for parallelization (if any).
//! - `cgns`: A correlated Gaussian noise system that produces two correlated Brownian increments.

use ndarray::Array1;

use crate::stochastic::noise::cgns::CGNS;
use crate::stochastic::FloatExt;
use crate::stochastic::ProcessExt;

/// Standard Duffie–Kan two-factor model (continuous, no jumps).
pub struct DuffieKan<T: FloatExt> {
  pub alpha: T,
  pub beta: T,
  pub gamma: T,
  pub rho: T,
  pub a1: T,
  pub b1: T,
  pub c1: T,
  pub sigma1: T,
  pub a2: T,
  pub b2: T,
  pub c2: T,
  pub sigma2: T,
  pub n: usize,
  pub r0: Option<T>,
  pub x0: Option<T>,
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
