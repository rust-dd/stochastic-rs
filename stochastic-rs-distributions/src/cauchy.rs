//! # Cauchy
//!
//! $$
//! f(x)=\frac{1}{\pi\gamma\left[1+\left(\frac{x-x_0}{\gamma}\right)^2\right]}
//! $$
//!
use std::cell::Cell;
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use super::SimdFloatExt;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;

const SMALL_CAUCHY_THRESHOLD: usize = 16;

pub struct SimdCauchy<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  x0: T,
  gamma: T,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<R>,
  stream_seed: Cell<u64>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdCauchy<T, R> {
  /// Creates a Cauchy distribution.
  ///
  /// - `x0` — location x₀ (matches the module header's x₀; also the
  ///   median and mode).
  /// - `gamma` — scale γ > 0 (matches the module header's γ; the
  ///   half-width at half-maximum).
  pub fn new<S: crate::simd_rng::SeedExt>(x0: T, gamma: T, seed: &S) -> Self {
    assert!(
      gamma > T::zero(),
      "gamma must satisfy `gamma > T::zero()`, got gamma = {gamma:?}"
    );
    let stream_seed = seed.seed_value();
    Self {
      x0,
      gamma,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      simd_rng: UnsafeCell::new(R::from_seed(stream_seed)),
      stream_seed: Cell::new(stream_seed),
    }
  }

  /// Builds an independent worker stream for the `stream_idx`-th chunk of a
  /// parallel `sample_matrix` fan-out; see
  /// [`DistributionSampler::fork`](crate::traits::DistributionSampler::fork).
  #[doc(hidden)]
  pub fn fork(&self, stream_idx: u64) -> Self {
    let mut basis = self.stream_seed.get();
    let call_basis = crate::simd_rng::derive_seed(&mut basis);
    self.stream_seed.set(basis);
    let child_seed = crate::simd_rng::derive_fork_seed(call_basis, stream_idx);
    Self::new(
      self.x0,
      self.gamma,
      &crate::simd_rng::Deterministic::new(child_seed),
    )
  }

  /// Returns a single sample using the internal SIMD RNG.
  #[inline]
  pub fn sample_fast(&self) -> T {
    let index = unsafe { &mut *self.index.get() };
    if *index >= 16 {
      self.refill_buffer();
    }
    let buf = unsafe { &mut *self.buffer.get() };
    let z = buf[*index];
    *index += 1;
    z
  }

  /// Fills `out` using the internal SIMD RNG stream — the only stream this
  /// sampler draws from (see the crate-level RNG policy).
  pub fn fill_slice(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    if out.len() < SMALL_CAUCHY_THRESHOLD {
      let pi = T::pi();
      let half = T::from(0.5).unwrap();
      for x in out.iter_mut() {
        let u = T::sample_uniform_simd(rng);
        *x = self.x0 + self.gamma * (pi * (u - half)).tan();
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

  fn refill_buffer(&self) {
    let buf = unsafe { &mut *self.buffer.get() };
    self.fill_slice(buf);
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdCauchy<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.x0, self.gamma, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdCauchy<T, R> {
  /// The `rng` argument is intentionally unused — this type draws from its
  /// own internal SIMD stream seeded at construction. Use `Deterministic`
  /// in the constructor for reproducibility.
  fn sample<Rr: Rng + ?Sized>(&self, _rng: &mut Rr) -> T {
    let idx = unsafe { &mut *self.index.get() };
    if *idx >= 16 {
      self.refill_buffer();
    }
    let val = unsafe { (*self.buffer.get())[*idx] };
    *idx += 1;
    val
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> crate::traits::DistributionExt for SimdCauchy<T, R> {
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
  use crate::traits::DistributionExt;

  /// Backs the doc comments on `mean`/`variance`/`skewness`/`kurtosis`/
  /// `moment_generating_function`: every one of these must actually return
  /// the claimed non-finite value, not just carry prose asserting it does.
  #[test]
  fn cauchy_moments_are_non_finite_as_documented() {
    let c = SimdCauchy::<f64>::new(1.5, 2.0, &Unseeded);
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
}
