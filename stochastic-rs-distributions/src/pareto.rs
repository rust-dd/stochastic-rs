//! # Pareto
//!
//! $$
//! f(x)=\alpha x_m^\alpha x^{-(\alpha+1)},\ x\ge x_m
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

const SMALL_PARETO_THRESHOLD: usize = 16;

pub struct SimdPareto<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  x_m: T,
  alpha: T,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<R>,
  stream_seed: Cell<u64>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdPareto<T, R> {
  /// Creates a Pareto (Type I) distribution.
  ///
  /// - `x_m` — minimum/scale x_m > 0 (matches the module header's x_m);
  ///   also the mode.
  /// - `alpha` — tail index α > 0 (matches the module header's α);
  ///   controls which moments exist (mean requires α>1, variance α>2).
  pub fn new<S: crate::simd_rng::SeedExt>(x_m: T, alpha: T, seed: &S) -> Self {
    assert!(
      x_m > T::zero() && alpha > T::zero(),
      "x_m must satisfy `x_m > T::zero() && alpha > T::zero()`, got x_m = {x_m:?}, alpha = {alpha:?}"
    );
    let stream_seed = seed.seed_value();
    Self {
      x_m,
      alpha,
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
      self.x_m,
      self.alpha,
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
    if out.len() < SMALL_PARETO_THRESHOLD {
      let neg_inv_alpha = -T::one() / self.alpha;
      let eps = T::min_positive_val();
      for x in out.iter_mut() {
        let u = T::sample_uniform_simd(rng);
        let base = (T::one() - u).max(eps);
        *x = self.x_m * (base.ln() * neg_inv_alpha).exp();
      }
      return;
    }
    let xm = T::splat(self.x_m);
    let neg_inv_alpha = T::splat(-T::one() / self.alpha);
    let one = T::splat(T::one());
    let eps = T::splat(T::min_positive_val());
    let mut u = [T::zero(); 8];
    let (chunks, rem) = out.as_chunks_mut::<8>();
    for chunk in chunks {
      T::fill_uniform_simd(rng, &mut u);
      let v = T::simd_from_array(u);
      let base = T::simd_max(one - v, eps);
      let x = xm * T::simd_exp(T::simd_ln(base) * neg_inv_alpha);
      *chunk = T::simd_to_array(x);
    }
    if !rem.is_empty() {
      T::fill_uniform_simd(rng, &mut u);
      let v = T::simd_from_array(u);
      let base = T::simd_max(one - v, eps);
      let x = T::simd_to_array(xm * T::simd_exp(T::simd_ln(base) * neg_inv_alpha));
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

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdPareto<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.x_m, self.alpha, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdPareto<T, R> {
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

impl<T: SimdFloatExt, R: SimdRngExt> crate::traits::DistributionExt for SimdPareto<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    let xm = self.x_m.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    if x < xm {
      0.0
    } else {
      a * xm.powf(a) / x.powf(a + 1.0)
    }
  }

  fn cdf(&self, x: f64) -> f64 {
    let xm = self.x_m.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    if x < xm { 0.0 } else { 1.0 - (xm / x).powf(a) }
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    let xm = self.x_m.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    xm / (1.0 - p).powf(1.0 / a)
  }

  /// `+∞`, not `NaN`, at `alpha <= 1`: the mean integral `∫x·f(x)dx`
  /// diverges to a definite (infinite) value at these — commonly used —
  /// shape parameters (e.g. the classic "80/20" Pareto has `alpha ≈ 1.16`).
  fn mean(&self) -> f64 {
    let xm = self.x_m.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    if a > 1.0 {
      xm * a / (a - 1.0)
    } else {
      f64::INFINITY
    }
  }

  fn median(&self) -> f64 {
    let xm = self.x_m.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    xm * 2.0_f64.powf(1.0 / a)
  }

  fn mode(&self) -> f64 {
    self.x_m.to_f64().unwrap()
  }

  /// `+∞`, not `NaN`, at `alpha <= 2`: same divergent-integral reason as
  /// `mean`, one moment order up.
  fn variance(&self) -> f64 {
    let xm = self.x_m.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    if a > 2.0 {
      xm * xm * a / ((a - 1.0).powi(2) * (a - 2.0))
    } else {
      f64::INFINITY
    }
  }

  /// `NaN`, not `+∞`, at `alpha <= 3`: unlike mean/variance, the third
  /// central moment does not merely diverge to a signed infinity here — it
  /// is not defined at all, so `NaN` is the honest answer rather than a
  /// sign choice.
  fn skewness(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    if a > 3.0 {
      2.0 * (1.0 + a) / (a - 3.0) * ((a - 2.0) / a).sqrt()
    } else {
      f64::NAN
    }
  }

  /// `NaN` at `alpha <= 4`, for the same reason as `skewness` one moment
  /// order up.
  fn kurtosis(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    if a > 4.0 {
      6.0 * (a.powi(3) + a.powi(2) - 6.0 * a - 2.0) / (a * (a - 3.0) * (a - 4.0))
    } else {
      f64::NAN
    }
  }

  fn entropy(&self) -> f64 {
    let xm = self.x_m.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    (xm / a).ln() + 1.0 / a + 1.0
  }

  /// `NaN` for every `t > 0`: the Pareto tail decays only polynomially
  /// (`~x^{-alpha-1}`), too slowly for `e^{tx}` to be integrable at any
  /// positive `t`, regardless of `alpha`.
  fn moment_generating_function(&self, _t: f64) -> f64 {
    f64::NAN
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::DistributionExt;

  /// Backs the doc comments on `mean`/`variance`/`skewness`/`kurtosis`:
  /// `alpha = 1.16` (the classic "80/20" Pareto) is a common, valid,
  /// in-range shape parameter, yet already sits below every one of these
  /// thresholds — mean is the only finite moment it has.
  #[test]
  fn pareto_80_20_moments_match_documented_thresholds() {
    let p = SimdPareto::<f64>::new(1.0, 1.16, &Unseeded);
    assert!(
      p.mean().is_finite(),
      "alpha=1.16 > 1, mean should be finite"
    );
    assert_eq!(p.variance(), f64::INFINITY, "alpha=1.16 <= 2");
    assert!(p.skewness().is_nan(), "alpha=1.16 <= 3");
    assert!(p.kurtosis().is_nan(), "alpha=1.16 <= 4");
    assert!(
      p.moment_generating_function(0.5).is_nan(),
      "MGF at t > 0 must be NaN"
    );
  }

  /// Above every threshold, all four moments must be finite real numbers.
  #[test]
  fn pareto_high_alpha_moments_are_all_finite() {
    let p = SimdPareto::<f64>::new(1.0, 5.0, &Unseeded);
    assert!(p.mean().is_finite());
    assert!(p.variance().is_finite());
    assert!(p.skewness().is_finite());
    assert!(p.kurtosis().is_finite());
  }
}

py_distribution!(PyPareto, SimdPareto,
  sig: (x_m, alpha, seed=None, dtype=None),
  params: (x_m: f64, alpha: f64)
);
