use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::uniform::SimdUniform;

use super::sample_positive_stable;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Alpha-stable subordinator with Laplace exponent `phi(lambda) = c * lambda^alpha`.
pub struct AlphaStableSubordinator<T: FloatExt, S: SeedExt = Unseeded> {
  /// Stability index in `(0, 1)`.
  pub alpha: T,
  /// Laplace scale coefficient `c > 0`.
  pub c: T,
  /// Number of grid points.
  pub n: usize,
  /// Initial level.
  pub x0: Option<T>,
  /// Horizon `T`; defaults to `1`.
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt> AlphaStableSubordinator<T> {
  pub fn new(alpha: T, c: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    assert!(
      alpha > T::zero() && alpha < T::one(),
      "alpha must be in (0,1)"
    );
    assert!(c > T::zero(), "c must be positive");
    Self {
      alpha,
      c,
      n,
      x0,
      t,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> AlphaStableSubordinator<T, Deterministic> {
  pub fn seeded(alpha: T, c: T, n: usize, x0: Option<T>, t: Option<T>, seed: u64) -> Self {
    assert!(
      alpha > T::zero() && alpha < T::one(),
      "alpha must be in (0,1)"
    );
    assert!(c > T::zero(), "c must be positive");
    Self {
      alpha,
      c,
      n,
      x0,
      t,
      seed: Deterministic::new(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for AlphaStableSubordinator<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let mut path = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return path;
    }
    path[0] = self.x0.unwrap_or(T::zero());
    if self.n == 1 {
      return path;
    }

    let t_max = self.t.unwrap_or(T::one()).to_f64().unwrap();
    let alpha = self.alpha.to_f64().unwrap();
    let c = self.c.to_f64().unwrap();
    let dt = t_max / (self.n - 1) as f64;
    let scale = (c * dt).powf(1.0 / alpha);
    let uniform = SimdUniform::<f64>::from_seed_source(0.0, 1.0, &self.seed);
    let mut level = path[0].to_f64().unwrap();

    for i in 1..self.n {
      let s = sample_positive_stable(alpha, &uniform);
      level += scale * s;
      path[i] = T::from_f64_fast(level);
    }

    path
  }
}

py_process_1d!(PyAlphaStableSubordinator, AlphaStableSubordinator,
  sig: (alpha, c, n, x0=None, t=None, seed=None, dtype=None),
  params: (alpha: f64, c: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
