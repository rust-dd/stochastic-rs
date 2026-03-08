//! # Fouque
//!
//! $$
//! dX_t=a(t,X_t)dt+b(t,X_t)dW_t
//! $$
//!
use ndarray::Array1;

use crate::simd_rng::Deterministic;
use crate::simd_rng::SeedExt;
use crate::simd_rng::Unseeded;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Fouque slow–fast OU system
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

impl<T: FloatExt> FouqueOU2D<T> {
  pub fn new(
    kappa: T,
    theta: T,
    epsilon: T,
    alpha: T,
    n: usize,
    x0: Option<T>,
    y0: Option<T>,
    t: Option<T>,
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
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> FouqueOU2D<T, Deterministic> {
  pub fn seeded(
    kappa: T,
    theta: T,
    epsilon: T,
    alpha: T,
    n: usize,
    x0: Option<T>,
    y0: Option<T>,
    t: Option<T>,
    seed: u64,
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
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for FouqueOU2D<T, S> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> [Array1<T>; 2] {
    let mut seed = self.seed;
    let mut x = Array1::<T>::zeros(self.n);
    let mut y = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return [x, y];
    }

    x[0] = self.x0.unwrap_or(T::zero());
    y[0] = self.y0.unwrap_or(T::zero());
    if self.n == 1 {
      return [x, y];
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let sqrt_dt = dt.sqrt();
    let mut gn_x = vec![T::zero(); n_increments];
    let mut gn_y = vec![T::zero(); n_increments];

    let nx = crate::distributions::normal::SimdNormal::<T>::from_seed_source(
      T::zero(),
      sqrt_dt,
      &mut seed,
    );
    let ny = crate::distributions::normal::SimdNormal::<T>::from_seed_source(
      T::zero(),
      sqrt_dt,
      &mut seed,
    );
    nx.fill_slice_fast(&mut gn_x);
    ny.fill_slice_fast(&mut gn_y);

    let eps = self.epsilon;
    let sqrt_eps_inv = T::one() / eps.sqrt();
    let eps_inv = T::one() / eps;

    for i in 1..self.n {
      // Slow OU
      x[i] = x[i - 1] + self.kappa * (self.theta - x[i - 1]) * dt + eps * gn_x[i - 1];
      // Fast OU
      y[i] = y[i - 1] + eps_inv * (self.alpha - y[i - 1]) * dt + sqrt_eps_inv * gn_y[i - 1];
    }

    [x, y]
  }
}

py_process_2x1d!(PyFouqueOU2D, FouqueOU2D,
  sig: (kappa, theta, epsilon, alpha, n, x0=None, y0=None, t=None, dtype=None),
  params: (kappa: f64, theta: f64, epsilon: f64, alpha: f64, n: usize, x0: Option<f64>, y0: Option<f64>, t: Option<f64>)
);
