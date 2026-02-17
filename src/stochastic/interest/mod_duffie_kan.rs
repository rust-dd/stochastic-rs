//! # Mod Duffie Kan
//!
//! $$
//! dX_t=K(\Theta-X_t)dt+\sqrt{A+BX_t}\,dW_t,\quad r_t=\ell_0+\ell^\top X_t
//! $$
//!
use ndarray::Array1;
use rand_distr::Distribution;

use crate::distributions::exp::SimdExp;
use crate::distributions::normal::SimdNormal;
use crate::stochastic::noise::cgns::CGNS;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct DuffieKanJumpExp<T: FloatExt> {
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
  /// Correlated Gaussian noise generator for the diffusion part.
  cgns: CGNS<T>,
}

impl<T: FloatExt> DuffieKanJumpExp<T> {
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
      cgns: CGNS::new(rho, n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for DuffieKanJumpExp<T> {
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
    let jump_dist = SimdNormal::<T, 64>::new(T::zero(), self.jump_scale);

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
        let jump_x = jump_dist.sample(&mut rng);
        jump_sum_x += jump_x;
        next_jump_time += exp_dist.sample(&mut rng);
      }

      r[i] = r_old + dr;
      x[i] = x_old + dx + jump_sum_x;
    }

    [r, x]
  }
}

py_process_2x1d!(PyDuffieKanJumpExp, DuffieKanJumpExp,
  sig: (alpha, beta, gamma_, rho, a1, b1, c1, sigma1, a2, b2, c2, sigma2, lambda_, jump_scale, n, r0=None, x0=None, t=None, dtype=None),
  params: (alpha: f64, beta: f64, gamma_: f64, rho: f64, a1: f64, b1: f64, c1: f64, sigma1: f64, a2: f64, b2: f64, c2: f64, sigma2: f64, lambda_: f64, jump_scale: f64, n: usize, r0: Option<f64>, x0: Option<f64>, t: Option<f64>)
);