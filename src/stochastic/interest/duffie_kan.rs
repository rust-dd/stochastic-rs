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

use impl_new_derive::ImplNew;
use ndarray::Array1;

use crate::stochastic::{noise::cgns::CGNS, Sampling2D};

/// Standard Duffie–Kan two-factor model (continuous, no jumps).
#[derive(ImplNew)]
pub struct DuffieKan {
  pub alpha: f64,
  pub beta: f64,
  pub gamma: f64,
  pub rho: f64,
  pub a1: f64,
  pub b1: f64,
  pub c1: f64,
  pub sigma1: f64,
  pub a2: f64,
  pub b2: f64,
  pub c2: f64,
  pub sigma2: f64,
  pub n: usize,
  pub r0: Option<f64>,
  pub x0: Option<f64>,
  pub t: Option<f64>,
  pub m: Option<usize>,
  pub cgns: CGNS,
}

impl Sampling2D<f64> for DuffieKan {
  /// Sample the Duffie-Kan process
  fn sample(&self) -> [Array1<f64>; 2] {
    let [cgn1, cgn2] = self.cgns.sample();
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;

    let mut r = Array1::<f64>::zeros(self.n);
    let mut x = Array1::<f64>::zeros(self.n);

    r[0] = self.r0.unwrap_or(0.0);
    x[0] = self.x0.unwrap_or(0.0);

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

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }

  /// Number of samples for parallel sampling
  fn m(&self) -> Option<usize> {
    self.m
  }
}
