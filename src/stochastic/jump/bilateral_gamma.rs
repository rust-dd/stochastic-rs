//! # BilateralGamma
//!
//! $$
//! X_t=\Gamma^+_t-\Gamma^-_t,\quad \Gamma^+\sim\Gamma(\alpha_p t,\lambda_p^{-1}),\ \Gamma^-\sim\Gamma(\alpha_m t,\lambda_m^{-1})
//! $$
//!
use ndarray::Array1;

use crate::distributions::gamma::SimdGamma;
use crate::distributions::normal::SimdNormal;
use crate::simd_rng::Deterministic;
use crate::simd_rng::SeedExt;
use crate::simd_rng::Unseeded;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Bilateral Gamma process.
///
/// Decomposes log-returns into independent positive and negative gamma jump
/// components, each with its own shape and rate parameter.
///
/// Characteristic exponent:
/// $$
/// \psi(\xi)=\alpha_p\ln\!\frac{\lambda_p}{\lambda_p-i\xi}+\alpha_m\ln\!\frac{\lambda_m}{\lambda_m+i\xi}
/// $$
pub struct BilateralGamma<T: FloatExt, S: SeedExt = Unseeded> {
  /// Shape parameter for positive jumps.
  pub alpha_p: T,
  /// Rate parameter for positive jumps.
  pub lambda_p: T,
  /// Shape parameter for negative jumps.
  pub alpha_m: T,
  /// Rate parameter for negative jumps.
  pub lambda_m: T,
  /// Number of discrete simulation points.
  pub n: usize,
  /// Initial value of the process.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1).
  pub t: Option<T>,
  pub seed: S,
}

impl<T: FloatExt> BilateralGamma<T> {
  pub fn new(
    alpha_p: T,
    lambda_p: T,
    alpha_m: T,
    lambda_m: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
  ) -> Self {
    assert!(alpha_p > T::zero(), "alpha_p must be positive");
    assert!(lambda_p > T::zero(), "lambda_p must be positive");
    assert!(alpha_m > T::zero(), "alpha_m must be positive");
    assert!(lambda_m > T::zero(), "lambda_m must be positive");
    Self {
      alpha_p,
      lambda_p,
      alpha_m,
      lambda_m,
      n,
      x0,
      t,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> BilateralGamma<T, Deterministic> {
  pub fn seeded(
    alpha_p: T,
    lambda_p: T,
    alpha_m: T,
    lambda_m: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: u64,
  ) -> Self {
    assert!(alpha_p > T::zero(), "alpha_p must be positive");
    assert!(lambda_p > T::zero(), "lambda_p must be positive");
    assert!(alpha_m > T::zero(), "alpha_m must be positive");
    assert!(lambda_m > T::zero(), "lambda_m must be positive");
    Self {
      alpha_p,
      lambda_p,
      alpha_m,
      lambda_m,
      n,
      x0,
      t,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> BilateralGamma<T, S> {
  #[inline]
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for BilateralGamma<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut x = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return x;
    }
    x[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return x;
    }

    let dt = self.dt();
    let mut seed = self.seed;

    // Gamma(shape = alpha * dt, scale = 1/lambda)
    let gamma_p =
      SimdGamma::from_seed_source(self.alpha_p * dt, T::one() / self.lambda_p, &mut seed);
    let mut gp = Array1::<T>::zeros(self.n - 1);
    gamma_p.fill_slice_fast(gp.as_slice_mut().unwrap());

    let gamma_m =
      SimdGamma::from_seed_source(self.alpha_m * dt, T::one() / self.lambda_m, &mut seed);
    let mut gm = Array1::<T>::zeros(self.n - 1);
    gamma_m.fill_slice_fast(gm.as_slice_mut().unwrap());

    for i in 1..self.n {
      x[i] = x[i - 1] + gp[i - 1] - gm[i - 1];
    }

    x
  }
}

/// Bilateral Gamma Motion: Brownian component plus Bilateral Gamma jumps.
///
/// $$
/// X_t=\sigma W_t+\Gamma^+_t-\Gamma^-_t
/// $$
pub struct BilateralGammaMotion<T: FloatExt, S: SeedExt = Unseeded> {
  /// Diffusion coefficient of the Brownian component.
  pub sigma: T,
  /// Shape parameter for positive jumps.
  pub alpha_p: T,
  /// Rate parameter for positive jumps.
  pub lambda_p: T,
  /// Shape parameter for negative jumps.
  pub alpha_m: T,
  /// Rate parameter for negative jumps.
  pub lambda_m: T,
  /// Number of discrete simulation points.
  pub n: usize,
  /// Initial value of the process.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1).
  pub t: Option<T>,
  pub seed: S,
}

impl<T: FloatExt> BilateralGammaMotion<T> {
  pub fn new(
    sigma: T,
    alpha_p: T,
    lambda_p: T,
    alpha_m: T,
    lambda_m: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
  ) -> Self {
    assert!(alpha_p > T::zero(), "alpha_p must be positive");
    assert!(lambda_p > T::zero(), "lambda_p must be positive");
    assert!(alpha_m > T::zero(), "alpha_m must be positive");
    assert!(lambda_m > T::zero(), "lambda_m must be positive");
    Self {
      sigma,
      alpha_p,
      lambda_p,
      alpha_m,
      lambda_m,
      n,
      x0,
      t,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> BilateralGammaMotion<T, Deterministic> {
  pub fn seeded(
    sigma: T,
    alpha_p: T,
    lambda_p: T,
    alpha_m: T,
    lambda_m: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: u64,
  ) -> Self {
    assert!(alpha_p > T::zero(), "alpha_p must be positive");
    assert!(lambda_p > T::zero(), "lambda_p must be positive");
    assert!(alpha_m > T::zero(), "alpha_m must be positive");
    assert!(lambda_m > T::zero(), "lambda_m must be positive");
    Self {
      sigma,
      alpha_p,
      lambda_p,
      alpha_m,
      lambda_m,
      n,
      x0,
      t,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> BilateralGammaMotion<T, S> {
  #[inline]
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for BilateralGammaMotion<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut x = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return x;
    }
    x[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return x;
    }

    let dt = self.dt();
    let sqrt_dt = dt.sqrt();
    let mut seed = self.seed;

    let gamma_p =
      SimdGamma::from_seed_source(self.alpha_p * dt, T::one() / self.lambda_p, &mut seed);
    let mut gp = Array1::<T>::zeros(self.n - 1);
    gamma_p.fill_slice_fast(gp.as_slice_mut().unwrap());

    let gamma_m =
      SimdGamma::from_seed_source(self.alpha_m * dt, T::one() / self.lambda_m, &mut seed);
    let mut gm = Array1::<T>::zeros(self.n - 1);
    gamma_m.fill_slice_fast(gm.as_slice_mut().unwrap());

    let normal = SimdNormal::<T>::from_seed_source(T::zero(), T::one(), &mut seed);
    let mut z = Array1::<T>::zeros(self.n - 1);
    normal.fill_slice_fast(z.as_slice_mut().unwrap());

    for i in 1..self.n {
      x[i] = x[i - 1] + self.sigma * sqrt_dt * z[i - 1] + gp[i - 1] - gm[i - 1];
    }

    x
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::ProcessExt;

  #[test]
  fn bg_n_eq_1_keeps_initial_value() {
    let p = BilateralGamma::new(1.0_f64, 2.0, 1.5, 2.5, 1, Some(3.0), Some(1.0));
    let x = p.sample();
    assert_eq!(x.len(), 1);
    assert_eq!(x[0], 3.0);
  }

  #[test]
  fn bgm_n_eq_1_keeps_initial_value() {
    let p = BilateralGammaMotion::new(0.2_f64, 1.0, 2.0, 1.5, 2.5, 1, Some(3.0), Some(1.0));
    let x = p.sample();
    assert_eq!(x.len(), 1);
    assert_eq!(x[0], 3.0);
  }

  #[test]
  fn bg_seeded_correct_length() {
    let p = BilateralGamma::seeded(1.0_f64, 2.0, 1.5, 2.5, 100, None, Some(1.0), 42);
    let x = p.sample();
    assert_eq!(x.len(), 100);
  }

  #[test]
  fn bgm_seeded_correct_length() {
    let p = BilateralGammaMotion::seeded(0.2_f64, 1.0, 2.0, 1.5, 2.5, 100, None, Some(1.0), 42);
    let x = p.sample();
    assert_eq!(x.len(), 100);
  }
}

py_process_1d!(PyBilateralGamma, BilateralGamma,
  sig: (alpha_p, lambda_p, alpha_m, lambda_m, n, x0=None, t=None, seed=None, dtype=None),
  params: (alpha_p: f64, lambda_p: f64, alpha_m: f64, lambda_m: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);

py_process_1d!(PyBilateralGammaMotion, BilateralGammaMotion,
  sig: (sigma, alpha_p, lambda_p, alpha_m, lambda_m, n, x0=None, t=None, seed=None, dtype=None),
  params: (sigma: f64, alpha_p: f64, lambda_p: f64, alpha_m: f64, lambda_m: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
