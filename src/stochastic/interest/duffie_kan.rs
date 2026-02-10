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

use crate::stochastic::noise::cgns::CGNS;
use crate::stochastic::Float;
use crate::stochastic::Process;

/// Standard Duffie–Kan two-factor model (continuous, no jumps).
#[derive(ImplNew)]
pub struct DuffieKan<T: Float> {
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
  pub m: Option<usize>,
  pub cgns: CGNS<T>,
}

impl<T: Float> Process<T> for DuffieKan<T> {
  type Output = [Array1<T>; 2];
  type Noise = CGNS<T>;

  fn sample(&self) -> Self::Output {
    self.euler_maruyama(|cgns| cgns.sample())
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Self::Output {
    self.euler_maruyama(|cgns| cgns.sample_simd())
  }

  fn euler_maruyama(
    &self,
    noise_fn: impl Fn(&Self::Noise) -> <Self::Noise as Process<T>>::Output,
  ) -> Self::Output {
    let cgns = CGNS::new(self.rho, self.n - 1, self.t);
    let dt = cgns.dt();
    let [cgn1, cgn2] = noise_fn(&cgns);

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
