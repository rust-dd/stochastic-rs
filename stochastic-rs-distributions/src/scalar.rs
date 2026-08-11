//! # Stateless scalar distributions
//!
//! The `Simd*` distributions in this crate own an amortised sample buffer and
//! their own RNG behind [`UnsafeCell`](std::cell::UnsafeCell), which makes them
//! fast but deliberately **not** [`Sync`]. That rules them out anywhere a
//! process requires `D: Distribution<T> + Send + Sync` — the jump-size slot of
//! [`CompoundPoisson`], `Bates1996`, `LevyDiffusion`, `JumpFOUCustom` and
//! friends, all of which inherit the bound from `ProcessExt: Send + Sync`.
//!
//! The types here fill that slot. They hold parameters only, draw from the
//! caller's RNG, and are therefore `Copy + Send + Sync`. Those processes
//! already build their own seeded RNG before sampling, so nothing is lost:
//! the stream still comes from the workspace seed source.
//!
//! Use `Simd*` for bulk fills you drive yourself; use these where a process
//! asks for a distribution it will drive.
//!
//! [`CompoundPoisson`]: https://docs.rs/stochastic-rs-stochastic
use rand::Rng;
use rand_distr::Distribution;

use crate::special::ndtri;
use crate::traits::FloatExt;

/// Normal distribution sampled by inverse CDF from the caller's RNG.
///
/// Exact — [`ndtri`] is the inverse of the standard normal CDF — at the cost
/// of one `ndtri` evaluation per draw, which costs roughly 1.8x a
/// [`SimdNormal`](crate::normal::SimdNormal) draw. Prefer `SimdNormal` for
/// bulk work you drive yourself, and this where `Sync` is required.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarNormal<T> {
  mean: T,
  std_dev: T,
}

impl<T: FloatExt> ScalarNormal<T> {
  /// Creates a stateless normal distribution.
  ///
  /// - `mean` — location μ.
  /// - `std_dev` — scale σ > 0.
  ///
  /// # Panics
  /// Panics if `std_dev` is not strictly positive.
  #[inline]
  pub fn new(mean: T, std_dev: T) -> Self {
    assert!(std_dev > T::zero(), "std_dev must be > 0");
    Self { mean, std_dev }
  }

  /// Mean μ — the location the `ndtri`-based inverse-CDF draw is
  /// centered on.
  #[inline]
  pub fn mean(&self) -> T {
    self.mean
  }

  /// Standard deviation σ > 0 — scales the `ndtri`-based inverse-CDF
  /// draw.
  #[inline]
  pub fn std_dev(&self) -> T {
    self.std_dev
  }
}

impl<T: FloatExt> Distribution<T> for ScalarNormal<T> {
  #[inline]
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
    // Clamped away from the open ends so `ndtri` never sees 0 or 1.
    let u = rng.random::<f64>().clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    self.mean + self.std_dev * T::from_f64_fast(ndtri(u))
  }
}

/// Exponential distribution sampled by inverse CDF from the caller's RNG.
///
/// $F^{-1}(u) = -\ln(1-u)/\lambda$. Exact, stateless, and `Sync`. The
/// logarithm costs roughly 2.8x a [`SimdExp`](crate::exp::SimdExp) draw,
/// whose ziggurat avoids it on about 98% of samples — a wider gap than the
/// normal case, so prefer `SimdExp` whenever `Sync` is not required.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarExp<T> {
  lambda: T,
}

impl<T: FloatExt> ScalarExp<T> {
  /// Creates a stateless exponential distribution.
  ///
  /// - `lambda` — rate λ > 0 (mean = 1/λ).
  ///
  /// # Panics
  /// Panics if `lambda` is not strictly positive.
  #[inline]
  pub fn new(lambda: T) -> Self {
    assert!(lambda > T::zero(), "lambda must be > 0");
    Self { lambda }
  }

  /// Rate λ (mean = 1/λ); consumed via `-ln(1-U)/λ` on each draw.
  #[inline]
  pub fn lambda(&self) -> T {
    self.lambda
  }
}

impl<T: FloatExt> Distribution<T> for ScalarExp<T> {
  #[inline]
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
    let u = rng.random::<f64>().clamp(0.0, 1.0 - f64::EPSILON);
    T::from_f64_fast(-(1.0 - u).ln()) / self.lambda
  }
}

#[cfg(test)]
mod tests {
  use rand_distr::Distribution;
  use stochastic_rs_core::simd_rng::SimdRng;

  use super::ScalarExp;
  use super::ScalarNormal;

  /// Both types must satisfy the `Send + Sync` bound that the jump processes
  /// impose — this is the whole reason they exist.
  #[test]
  fn scalar_distributions_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ScalarNormal<f64>>();
    assert_send_sync::<ScalarExp<f64>>();
  }

  #[test]
  fn scalar_normal_moments_match() {
    let d = ScalarNormal::<f64>::new(-0.75, 1.35);
    let mut rng = SimdRng::from_seed(42);
    let n = 200_000;
    let xs = (0..n).map(|_| d.sample(&mut rng)).collect::<Vec<_>>();

    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    let se = 1.35 / (n as f64).sqrt();

    assert!((mean + 0.75).abs() < 6.0 * se, "mean = {mean}");
    assert!((var - 1.35 * 1.35).abs() < 0.05, "var = {var}");
  }

  #[test]
  fn scalar_exp_moments_match() {
    let lambda = 1.8_f64;
    let d = ScalarExp::<f64>::new(lambda);
    let mut rng = SimdRng::from_seed(7);
    let n = 200_000;
    let xs = (0..n).map(|_| d.sample(&mut rng)).collect::<Vec<_>>();

    assert!(
      xs.iter().all(|x| *x >= 0.0),
      "exp draws must be non-negative"
    );

    let mean = xs.iter().sum::<f64>() / n as f64;
    let se = (1.0 / lambda) / (n as f64).sqrt();
    assert!((mean - 1.0 / lambda).abs() < 6.0 * se, "mean = {mean}");
  }
}
