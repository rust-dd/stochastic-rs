use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::uniform::SimdUniform;

use super::sample_positive_stable;
use crate::buffer::array1_from_fill;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
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

impl<T: FloatExt, S: SeedExt> AlphaStableSubordinator<T, S> {
  pub fn new(alpha: T, c: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
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
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for AlphaStableSubordinator<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = AlphaStableSubordinatorSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> AlphaStableSubordinatorSampler<T> {
    let x0 = self.x0.unwrap_or(T::zero());
    let n_increments = self.n.saturating_sub(1).max(1);
    let t_max = self.t.unwrap_or(T::one()).to_f64().unwrap();
    let alpha = self.alpha.to_f64().unwrap();
    let c = self.c.to_f64().unwrap();
    let dt = t_max / n_increments as f64;
    let scale = (c * dt).powf(1.0 / alpha);
    AlphaStableSubordinatorSampler {
      n: self.n,
      x0,
      alpha,
      scale,
      uniform: SimdUniform::<f64>::new(0.0, 1.0, &self.seed),
    }
  }
}

/// Reusable [`AlphaStableSubordinator`] sampling state: the owned uniform
/// source driving the positive-stable increments plus precomputed scales.
#[doc(hidden)]
pub struct AlphaStableSubordinatorSampler<T: FloatExt> {
  n: usize,
  x0: T,
  alpha: f64,
  scale: f64,
  uniform: SimdUniform<f64>,
}

impl<T: FloatExt> AlphaStableSubordinatorSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let mut level = self.x0.to_f64().unwrap();
    for x in out[1..].iter_mut() {
      let s = sample_positive_stable(self.alpha, &self.uniform);
      level += self.scale * s;
      *x = T::from_f64_fast(level);
    }
  }
}

impl<T: FloatExt> PathSampler<T> for AlphaStableSubordinatorSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("AlphaStableSubordinator output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyAlphaStableSubordinator, AlphaStableSubordinator,
  sig: (alpha, c, n, x0=None, t=None, seed=None, dtype=None),
  params: (alpha: f64, c: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
