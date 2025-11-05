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

use crate::stochastic::{noise::cgns::CGNS, Sampling2DExt};
use impl_new_derive::ImplNew;
use ndarray::Array1;
use rand_distr::{Distribution, Exp, Normal};

#[derive(ImplNew)]
pub struct DuffieKanJumpExp<T> {
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

  /// Optional batch size.
  pub m: Option<usize>,

  /// Correlated Gaussian noise generator for the diffusion part.
  pub cgns: CGNS<T>,
}

#[cfg(feature = "f64")]
impl Sampling2DExt<f64> for DuffieKanJumpExp<f64> {
  fn sample(&self) -> [Array1<f64>; 2] {
    let [cgn1, cgn2] = self.cgns.sample();
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f64;
    let mut r = Array1::<f64>::zeros(self.n);
    let mut x = Array1::<f64>::zeros(self.n);
    r[0] = self.r0.unwrap_or(0.0);
    x[0] = self.x0.unwrap_or(0.0);

    let mut rng = rand::thread_rng();
    let exp_dist = Exp::new(self.lambda).unwrap();
    let jump_dist = Normal::new(0.0, self.jump_scale).unwrap();
    let mut next_jump_time = exp_dist.sample(&mut rng);

    for i in 1..self.n {
      let current_time = i as f64 * dt;
      let r_old = r[i - 1];
      let x_old = x[i - 1];

      let dr = (self.a1 * r_old + self.b1 * x_old + self.c1) * dt
        + self.sigma1 * (self.alpha * r_old + self.beta * x_old + self.gamma) * cgn1[i - 1];
      let dx = (self.a2 * r_old + self.b2 * x_old + self.c2) * dt
        + self.sigma2 * (self.alpha * r_old + self.beta * x_old + self.gamma) * cgn2[i - 1];

      let mut jump_sum_x = 0.0;
      while next_jump_time <= current_time {
        let jump_x = jump_dist.sample(&mut rng);
        jump_sum_x += jump_x;
        next_jump_time += exp_dist.sample(&mut rng);
      }

      r[i] = r_old + dr;
      x[i] = x_old + dx + jump_sum_x;
    }

    [r, x]
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(feature = "f32")]
impl Sampling2DExt<f32> for DuffieKanJumpExp<f32> {
  fn sample(&self) -> [Array1<f32>; 2] {
    let [cgn1, cgn2] = self.cgns.sample();
    let dt = self.t.unwrap_or(1.0) / (self.n - 1) as f32;
    let mut r = Array1::<f32>::zeros(self.n);
    let mut x = Array1::<f32>::zeros(self.n);
    r[0] = self.r0.unwrap_or(0.0);
    x[0] = self.x0.unwrap_or(0.0);

    let mut rng = rand::thread_rng();
    let exp_dist = Exp::new(self.lambda as f64).unwrap();
    let jump_dist = Normal::new(0.0, self.jump_scale as f64).unwrap();
    let mut next_jump_time = exp_dist.sample(&mut rng) as f32;

    for i in 1..self.n {
      let current_time = i as f32 * dt;
      let r_old = r[i - 1];
      let x_old = x[i - 1];

      let dr = (self.a1 * r_old + self.b1 * x_old + self.c1) * dt
        + self.sigma1 * (self.alpha * r_old + self.beta * x_old + self.gamma) * cgn1[i - 1];
      let dx = (self.a2 * r_old + self.b2 * x_old + self.c2) * dt
        + self.sigma2 * (self.alpha * r_old + self.beta * x_old + self.gamma) * cgn2[i - 1];

      let mut jump_sum_x = 0.0;
      while next_jump_time <= current_time {
        let jump_x = jump_dist.sample(&mut rng) as f32;
        jump_sum_x += jump_x;
        next_jump_time += exp_dist.sample(&mut rng) as f32;
      }

      r[i] = r_old + dr;
      x[i] = x_old + dx + jump_sum_x;
    }

    [r, x]
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}
