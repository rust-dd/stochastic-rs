//! # BilateralGamma
//!
//! $$
//! X_t=\Gamma^+_t-\Gamma^-_t,\quad \Gamma^+\sim\Gamma(\alpha_p t,\lambda_p^{-1}),\ \Gamma^-\sim\Gamma(\alpha_m t,\lambda_m^{-1})
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
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

impl<T: FloatExt, S: SeedExt> BilateralGamma<T, S> {
  pub fn new(
    alpha_p: T,
    lambda_p: T,
    alpha_m: T,
    lambda_m: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
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
      seed,
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
  type Sampler<'s>
    = BilateralGammaSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> BilateralGammaSampler<T> {
    // Positive and negative gamma sources are derived from `self.seed` in the
    // same order as the legacy `sample()`, so the first fill matches
    // bit-for-bit; both owned sources advance on reuse for independent paths.
    let dt = self.dt();
    BilateralGammaSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      gamma_p: SimdGamma::<T>::new(self.alpha_p * dt, T::one() / self.lambda_p, &self.seed),
      gamma_m: SimdGamma::<T>::new(self.alpha_m * dt, T::one() / self.lambda_m, &self.seed),
    }
  }
}

/// Reusable [`BilateralGamma`] sampling state: owns the positive- and
/// negative-side gamma sources so a Monte-Carlo loop pays their setup once.
#[doc(hidden)]
pub struct BilateralGammaSampler<T: FloatExt> {
  n: usize,
  x0: T,
  gamma_p: SimdGamma<T>,
  gamma_m: SimdGamma<T>,
}

impl<T: FloatExt> BilateralGammaSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }

    let mut gp = Array1::<T>::zeros(out.len() - 1);
    self.gamma_p.fill_slice_fast(gp.as_slice_mut().unwrap());
    let mut gm = Array1::<T>::zeros(out.len() - 1);
    self.gamma_m.fill_slice_fast(gm.as_slice_mut().unwrap());

    for i in 1..out.len() {
      out[i] = out[i - 1] + gp[i - 1] - gm[i - 1];
    }
  }
}

impl<T: FloatExt> PathSampler<T> for BilateralGammaSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(
      out
        .as_slice_mut()
        .expect("BilateralGamma output must be contiguous"),
    );
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
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

impl<T: FloatExt, S: SeedExt> BilateralGammaMotion<T, S> {
  pub fn new(
    sigma: T,
    alpha_p: T,
    lambda_p: T,
    alpha_m: T,
    lambda_m: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
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
      seed,
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
  type Sampler<'s>
    = BilateralGammaMotionSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> BilateralGammaMotionSampler<T> {
    // Gamma sources then the Gaussian source are derived from `self.seed` in
    // the same order as the legacy `sample()`, so the first fill matches
    // bit-for-bit; all owned sources advance on reuse for independent paths.
    let dt = self.dt();
    BilateralGammaMotionSampler {
      n: self.n,
      sigma: self.sigma,
      x0: self.x0.unwrap_or(T::zero()),
      sqrt_dt: dt.sqrt(),
      gamma_p: SimdGamma::<T>::new(self.alpha_p * dt, T::one() / self.lambda_p, &self.seed),
      gamma_m: SimdGamma::<T>::new(self.alpha_m * dt, T::one() / self.lambda_m, &self.seed),
      normal: SimdNormal::<T>::new(T::zero(), T::one(), &self.seed),
    }
  }
}

/// Reusable [`BilateralGammaMotion`] sampling state: owns both gamma sources
/// and the Gaussian source so a Monte-Carlo loop pays their setup once.
#[doc(hidden)]
pub struct BilateralGammaMotionSampler<T: FloatExt> {
  n: usize,
  sigma: T,
  x0: T,
  sqrt_dt: T,
  gamma_p: SimdGamma<T>,
  gamma_m: SimdGamma<T>,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> BilateralGammaMotionSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }

    let mut gp = Array1::<T>::zeros(out.len() - 1);
    self.gamma_p.fill_slice_fast(gp.as_slice_mut().unwrap());
    let mut gm = Array1::<T>::zeros(out.len() - 1);
    self.gamma_m.fill_slice_fast(gm.as_slice_mut().unwrap());
    let mut z = Array1::<T>::zeros(out.len() - 1);
    self.normal.fill_slice_fast(z.as_slice_mut().unwrap());

    for i in 1..out.len() {
      out[i] = out[i - 1] + self.sigma * self.sqrt_dt * z[i - 1] + gp[i - 1] - gm[i - 1];
    }
  }
}

impl<T: FloatExt> PathSampler<T> for BilateralGammaMotionSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(
      out
        .as_slice_mut()
        .expect("BilateralGammaMotion output must be contiguous"),
    );
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::traits::ProcessExt;

  #[test]
  fn bg_n_eq_1_keeps_initial_value() {
    let p = BilateralGamma::new(1.0_f64, 2.0, 1.5, 2.5, 1, Some(3.0), Some(1.0), Unseeded);
    let x = p.sample();
    assert_eq!(x.len(), 1);
    assert_eq!(x[0], 3.0);
  }

  #[test]
  fn bgm_n_eq_1_keeps_initial_value() {
    let p = BilateralGammaMotion::new(
      0.2_f64,
      1.0,
      2.0,
      1.5,
      2.5,
      1,
      Some(3.0),
      Some(1.0),
      Unseeded,
    );
    let x = p.sample();
    assert_eq!(x.len(), 1);
    assert_eq!(x[0], 3.0);
  }

  #[test]
  fn bg_seeded_correct_length() {
    let p = BilateralGamma::new(
      1.0_f64,
      2.0,
      1.5,
      2.5,
      100,
      None,
      Some(1.0),
      Deterministic::new(42),
    );
    let x = p.sample();
    assert_eq!(x.len(), 100);
  }

  #[test]
  fn bgm_seeded_correct_length() {
    let p = BilateralGammaMotion::new(
      0.2_f64,
      1.0,
      2.0,
      1.5,
      2.5,
      100,
      None,
      Some(1.0),
      Deterministic::new(42),
    );
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
