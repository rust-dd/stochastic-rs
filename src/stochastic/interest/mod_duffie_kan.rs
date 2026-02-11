//! A modified Duffie–Kan model with jumps (jump diffusion) where jumps occur at exponentially distributed times.
//!
//! # Model Definition
//!
//! In addition to the continuous part:
//!
//! \
//! \[
//!   dr(t) = (a_1 r(t) + b_1 x(t) + c_1)\,dt
//!           + \sigma_1 (α\,r(t) + β\,x(t) + γ)\,dW_1(t)
//! \]
//! \[
//!   dx(t) = (a_2 r(t) + b_2 x(t) + c_2)\,dt
//!           + \sigma_2 (α\,r(t) + β\,x(t) + γ)\,dW_2(t)
//!           + dJ_x(t),
//! \]
//!
//! where jumps occur at random times with inter-arrival times following an exponential distribution with rate `lambda`.
//! Jump sizes are drawn from a normal distribution with mean 0 and standard deviation `jump_scale`.
//!
//! # Parameters
//! - `alpha, beta, gamma, a1, b1, c1, sigma1, a2, b2, c2, sigma2`: Model parameters for the continuous part.
//! - `lambda`: Jump intensity (rate for the exponential waiting times).
//! - `jump_scale`: Standard deviation for the normally distributed jump sizes.
//! - `r0, x0`: Initial values for r(t) and x(t).
//! - `t`: Total time horizon.
//! - `n`: Number of time steps.
//! - `m`: Optional batch size for parallel sampling.
//! - `cgns`: Correlated Gaussian noise generator for the diffusion part.

use ndarray::Array1;
use rand_distr::Distribution;

use crate::distributions::exp::SimdExp;
use crate::distributions::normal::SimdNormal;
use crate::stochastic::noise::cgns::CGNS;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct DuffieKanJumpExp<T: Float> {
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
  /// Jump intensity (rate for the exponential distribution).
  pub lambda: T,
  /// Standard deviation for jump sizes.
  pub jump_scale: T,
  /// Number of time steps.
  pub n: usize,
  /// Initial value for r(t).
  pub r0: Option<T>,
  /// Initial value for x(t).
  pub x0: Option<T>,
  /// Total time horizon.
  pub t: Option<T>,
  /// Jump distribution.
  jump_dist: SimdNormal<T>,
  /// Correlated Gaussian noise generator for the diffusion part.
  cgns: CGNS<T>,
}

unsafe impl<T: Float> Send for DuffieKanJumpExp<T> {}
unsafe impl<T: Float> Sync for DuffieKanJumpExp<T> {}

impl<T: Float> DuffieKanJumpExp<T> {
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
    lambda: T,
    jump_scale: T,
    n: usize,
    r0: Option<T>,
    x0: Option<T>,
    t: Option<T>,
  ) -> Self {
    let jump_dist = SimdNormal::new(T::zero(), jump_scale);

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
      lambda,
      jump_scale,
      n,
      r0,
      x0,
      t,
      jump_dist,
      cgns: CGNS::new(rho, n - 1, t),
    }
  }
}

impl<T: Float> Process<T> for DuffieKanJumpExp<T> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let dt = self.cgns.dt();
    let [cgn1, cgn2] = &self.cgns.sample();

    let mut r = Array1::<T>::zeros(self.n);
    let mut x = Array1::<T>::zeros(self.n);
    r[0] = self.r0.unwrap_or(T::zero());
    x[0] = self.x0.unwrap_or(T::zero());

    let mut rng = rand::rng();
    let exp_dist = SimdExp::new(self.lambda);

    let mut next_jump_time = exp_dist.sample(&mut rng);

    for i in 1..self.n {
      let current_time = T::from_usize_(i) * dt;
      let r_old = r[i - 1];
      let x_old = x[i - 1];

      let dr = (self.a1 * r_old + self.b1 * x_old + self.c1) * dt
        + self.sigma1 * (self.alpha * r_old + self.beta * x_old + self.gamma) * cgn1[i - 1];
      let dx = (self.a2 * r_old + self.b2 * x_old + self.c2) * dt
        + self.sigma2 * (self.alpha * r_old + self.beta * x_old + self.gamma) * cgn2[i - 1];

      let mut jump_sum_x = T::zero();
      while next_jump_time <= current_time {
        let jump_x = self.jump_dist.sample(&mut rng);
        jump_sum_x += jump_x;
        next_jump_time += exp_dist.sample(&mut rng);
      }

      r[i] = r_old + dr;
      x[i] = x_old + dx + jump_sum_x;
    }

    [r, x]
  }
}
