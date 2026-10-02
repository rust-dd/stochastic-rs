//! # Cauchy
//!
//! $$
//! f(x)=\frac{1}{\pi\gamma\left[1+\left(\frac{x-x_0}{\gamma}\right)^2\right]}
//! $$
//!
//! Sampling: inversion, Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, §II.2, DOI 10.1007/978-1-4613-8643-8.

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use super::SimdFloatExt;
use crate::seeded::Buffered;
use crate::seeded::StreamState;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

const SMALL_CAUCHY_THRESHOLD: usize = 16;

/// Cauchy law with location `x0` and scale `gamma`: parameters only; a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdCauchy<T> {
  x0: T,
  gamma: T,
}

impl<T: SimdFloatExt> SimdCauchy<T> {
  /// Creates a Cauchy distribution.
  ///
  /// - `x0` — location x₀ (matches the module header's x₀; also the
  ///   median and mode).
  /// - `gamma` — scale γ > 0 (matches the module header's γ; the
  ///   half-width at half-maximum).
  pub fn new(x0: T, gamma: T) -> Self {
    assert!(
      gamma > T::zero(),
      "gamma must satisfy `gamma > T::zero()`, got gamma = {gamma:?}"
    );
    Self { x0, gamma }
  }

  /// The location `x₀`.
  pub fn x0(&self) -> T {
    self.x0
  }

  /// The scale `γ`.
  pub fn gamma(&self) -> T {
    self.gamma
  }

  /// `x0 + γ·tan(π(u − ½))`, the inverse cdf at `u`.
  #[inline]
  fn invert(&self, u: T) -> T {
    self.x0 + self.gamma * (T::pi() * (u - T::from(0.5).unwrap())).tan()
  }

  fn fill_parts<R: SimdRngExt>(&self, rng: &mut R, out: &mut [T]) {
    if out.len() < SMALL_CAUCHY_THRESHOLD {
      for x in out.iter_mut() {
        *x = self.invert(T::sample_uniform_simd(rng));
      }
      return;
    }
    let x0 = T::splat(self.x0);
    let g = T::splat(self.gamma);
    let pi = T::splat(T::pi());
    let half = T::splat(T::from(0.5).unwrap());
    let mut u = [T::zero(); 8];
    let (chunks, rem) = out.as_chunks_mut::<8>();
    for chunk in chunks {
      T::fill_uniform_simd(rng, &mut u);
      let v = T::simd_from_array(u);
      let z = T::simd_tan(pi * (v - half));
      let x = x0 + g * z;
      *chunk = T::simd_to_array(x);
    }
    if !rem.is_empty() {
      T::fill_uniform_simd(rng, &mut u);
      let v = T::simd_from_array(u);
      let z = T::simd_tan(pi * (v - half));
      let x = T::simd_to_array(x0 + g * z);
      rem.copy_from_slice(&x[..rem.len()]);
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.invert(T::sample_uniform(rng))
  }
}

impl<T: SimdFloatExt> Sealed for SimdCauchy<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdCauchy<T> {
  type State<R: SimdRngExt> = StreamState<T, R, 16>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (StreamState<T, R, 16>, u64) {
    let stream_seed = seed.next_seed();
    (
      StreamState {
        rng: R::from_seed(stream_seed),
        buf: Buffered::new(),
      },
      stream_seed,
    )
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdCauchy<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 16>, out: &mut [T]) {
    self.fill_parts(&mut state.rng, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 16>) -> T {
    let StreamState { rng, buf } = state;
    buf.pop(|b| self.fill_parts(rng, b))
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdCauchy<T> {
  /// The inverse cdf at one `[0, 1)` uniform from the caller's rng (53 bits for `f64`, 24 for `f32`).
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdCauchy<T> {
  fn pdf(&self, x: f64) -> f64 {
    let x0 = self.x0.to_f64().unwrap();
    let g = self.gamma.to_f64().unwrap();
    1.0 / (std::f64::consts::PI * g * (1.0 + ((x - x0) / g).powi(2)))
  }

  fn cdf(&self, x: f64) -> f64 {
    let x0 = self.x0.to_f64().unwrap();
    let g = self.gamma.to_f64().unwrap();
    0.5 + ((x - x0) / g).atan() / std::f64::consts::PI
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    let x0 = self.x0.to_f64().unwrap();
    let g = self.gamma.to_f64().unwrap();
    x0 + g * (std::f64::consts::PI * (p - 0.5)).tan()
  }

  /// `NaN`, not "unimplemented": the defining integral `∫x·f(x)dx` does not
  /// converge absolutely, so the Cauchy distribution provably has no mean —
  /// median/mode (both `x0`) are the location statistics to use instead.
  fn mean(&self) -> f64 {
    f64::NAN
  }

  fn median(&self) -> f64 {
    self.x0.to_f64().unwrap()
  }

  fn mode(&self) -> f64 {
    self.x0.to_f64().unwrap()
  }

  /// `+∞`, not `NaN`: unlike the mean, the variance integral diverges to a
  /// definite (infinite) value rather than failing to converge at all.
  fn variance(&self) -> f64 {
    f64::INFINITY
  }

  /// `NaN`: skewness is a ratio built from the (nonexistent) mean and a
  /// third central moment that itself does not converge.
  fn skewness(&self) -> f64 {
    f64::NAN
  }

  /// `NaN`: kurtosis is a ratio built from the (nonexistent) mean and a
  /// fourth central moment that itself does not converge.
  fn kurtosis(&self) -> f64 {
    f64::NAN
  }

  fn entropy(&self) -> f64 {
    let g = self.gamma.to_f64().unwrap();
    (4.0 * std::f64::consts::PI * g).ln()
  }

  fn characteristic_function(&self, t: f64) -> num_complex::Complex64 {
    // φ(t) = exp(it x₀ - γ |t|)
    let x0 = self.x0.to_f64().unwrap();
    let g = self.gamma.to_f64().unwrap();
    num_complex::Complex64::new(-g * t.abs(), t * x0).exp()
  }

  /// `NaN`: the MGF `E[e^{tX}]` diverges for every `t != 0` because the
  /// Cauchy tail decays only as `1/x^2`, too slowly for `e^{tx}` to be
  /// integrable — use `characteristic_function` instead, which exists for
  /// every Cauchy parameter.
  fn moment_generating_function(&self, _t: f64) -> f64 {
    f64::NAN
  }
}

py_distribution!(PyCauchy, SimdCauchy,
  sig: (x0, gamma_, seed=None, dtype=None),
  params: (x0: f64, gamma_: f64)
);

#[cfg(test)]
mod tests {
  use super::*;
  use crate::tests::scalar_ks_best_p;
  use crate::traits::DistributionExt;

  /// Backs the doc comments on `mean`/`variance`/`skewness`/`kurtosis`/
  /// `moment_generating_function`: every one of these must actually return
  /// the claimed non-finite value, not just carry prose asserting it does.
  #[test]
  fn cauchy_moments_are_non_finite_as_documented() {
    let c = SimdCauchy::<f64>::new(1.5, 2.0);
    assert!(c.mean().is_nan(), "mean must be NaN, got {}", c.mean());
    assert_eq!(c.variance(), f64::INFINITY, "variance must be +inf");
    assert!(c.skewness().is_nan(), "skewness must be NaN");
    assert!(c.kurtosis().is_nan(), "kurtosis must be NaN");
    assert!(
      c.moment_generating_function(0.5).is_nan(),
      "MGF at t != 0 must be NaN"
    );
    assert_eq!(c.median(), 1.5, "median must equal x0");
    assert_eq!(c.mode(), 1.5, "mode must equal x0");
  }

  /// The honest `Distribution` draws from the caller's rng and agrees with the cdf.
  #[test]
  fn scalar_sample_matches_cdf() {
    let d = SimdCauchy::<f64>::new(1.0, 0.5);
    let best = scalar_ks_best_p(&d, |x| d.cdf(x));
    assert!(best > 0.01, "best p = {best}");
  }
}
