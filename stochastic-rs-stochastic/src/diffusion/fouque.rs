//! # Fouque
//!
//! $$
//! dX_t=a(t,X_t)dt+b(t,X_t)dW_t
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Fouque slow–fast Ou system
///
/// dX_t = kappa (theta - X_t) dt + epsilon dW_t
/// dY_t = (1/epsilon) (alpha - Y_t) dt + (1/sqrt(epsilon)) dZ_t
pub struct FouqueOU2D<T: FloatExt, S: SeedExt = Unseeded> {
  /// Mean-reversion speed parameter.
  pub kappa: T,
  /// Long-run target level / model location parameter.
  pub theta: T,
  /// Model parameter controlling process dynamics.
  pub epsilon: T,
  /// Model shape / loading parameter.
  pub alpha: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Initial value of the secondary state variable.
  pub y0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> FouqueOU2D<T, S> {
  pub fn new(
    kappa: T,
    theta: T,
    epsilon: T,
    alpha: T,
    n: usize,
    x0: Option<T>,
    y0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(epsilon > T::zero(), "epsilon must be positive");

    Self {
      kappa,
      theta,
      epsilon,
      alpha,
      n,
      x0,
      y0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for FouqueOU2D<T, S> {
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = FouqueOU2DSampler<T, S>
  where
    Self: 's;

  fn sampler(&self) -> FouqueOU2DSampler<T, S> {
    FouqueOU2DSampler {
      kappa: self.kappa,
      theta: self.theta,
      epsilon: self.epsilon,
      alpha: self.alpha,
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      y0: self.y0.unwrap_or(T::zero()),
      t: self.t,
      seed: self.seed.clone(),
    }
  }
}

/// Reusable [`FouqueOU2D`] sampling state: owns the seed source so a Monte-Carlo
/// loop reuses both output buffers. The two Gaussian streams are rebuilt per
/// call from the derived seed, exactly as the legacy `sample` body did.
#[doc(hidden)]
pub struct FouqueOU2DSampler<T: FloatExt, S: SeedExt> {
  kappa: T,
  theta: T,
  epsilon: T,
  alpha: T,
  n: usize,
  x0: T,
  y0: T,
  t: Option<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> FouqueOU2DSampler<T, S> {
  fn fill_paths(&mut self, x: &mut [T], y: &mut [T]) {
    if self.n == 0 {
      return;
    }
    x[0] = self.x0;
    y[0] = self.y0;
    if self.n == 1 {
      return;
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let sqrt_dt = dt.sqrt();
    let mut gn_x = vec![T::zero(); n_increments];
    let mut gn_y = vec![T::zero(); n_increments];

    let nx = SimdNormal::<T>::new(T::zero(), sqrt_dt, &self.seed);
    let ny = SimdNormal::<T>::new(T::zero(), sqrt_dt, &self.seed);
    nx.fill_slice_fast(&mut gn_x);
    ny.fill_slice_fast(&mut gn_y);

    let eps = self.epsilon;
    let sqrt_eps_inv = T::one() / eps.sqrt();
    let eps_inv = T::one() / eps;

    for i in 1..self.n {
      // Slow Ou
      x[i] = x[i - 1] + self.kappa * (self.theta - x[i - 1]) * dt + eps * gn_x[i - 1];
      // Fast Ou
      y[i] = y[i - 1] + eps_inv * (self.alpha - y[i - 1]) * dt + sqrt_eps_inv * gn_y[i - 1];
    }
  }
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for FouqueOU2DSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [x, y] = out;
    self.fill_paths(
      x.as_slice_mut().expect("Fouque output must be contiguous"),
      y.as_slice_mut().expect("Fouque output must be contiguous"),
    );
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut x = Array1::<T>::zeros(self.n);
    let mut y = Array1::<T>::zeros(self.n);
    self.fill_paths(
      x.as_slice_mut().expect("contiguous"),
      y.as_slice_mut().expect("contiguous"),
    );
    [x, y]
  }
}

py_process_2x1d!(PyFouqueOU2D, FouqueOU2D,
  sig: (kappa, theta, epsilon, alpha, n, x0=None, y0=None, t=None, seed=None, dtype=None),
  params: (kappa: f64, theta: f64, epsilon: f64, alpha: f64, n: usize, x0: Option<f64>, y0: Option<f64>, t: Option<f64>)
);
