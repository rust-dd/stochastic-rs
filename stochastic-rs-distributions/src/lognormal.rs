//! # Lognormal
//!
//! $$
//! f(x)=\frac{1}{x\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(\ln x-\mu)^2}{2\sigma^2}\right),\ x>0
//! $$
//!
use std::cell::Cell;
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use super::SimdFloatExt;
use super::normal::SimdNormal;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;

pub struct SimdLogNormal<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  mu: T,
  sigma: T,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  normal: SimdNormal<T, 64, R>,
  stream_seed: Cell<u64>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdLogNormal<T, R> {
  /// Creates a log-normal distribution.
  ///
  /// - `mu` — mean of **ln(X)**, not of X itself (matches the module
  ///   header's μ; `mean() = exp(μ + σ²/2)` is the LogNormal-mean
  ///   formula derived from it).
  /// - `sigma` — standard deviation of **ln(X)**, not of X itself
  ///   (matches the module header's σ), must be > 0.
  ///
  /// RNGs come from a [`SeedExt`](crate::simd_rng::SeedExt) source.
  ///
  /// All sampling routes through the inner `normal`'s own stream (see
  /// [`Self::fill_slice`]), so this type has no separate engine of its
  /// own — `stream_seed` reuses `normal`'s already-captured value purely so
  /// [`Self::fork`] has a stable anchor to derive parallel worker streams
  /// from (see `SimdBeta::new` for the same reuse pattern). Its own
  /// independent `Cell`, so this fork cursor advances on its own from here.
  pub fn new<S: crate::simd_rng::SeedExt>(mu: T, sigma: T, seed: &S) -> Self {
    assert!(
      sigma > T::zero(),
      "sigma must satisfy `sigma > T::zero()`, got sigma = {sigma:?}"
    );
    let normal = SimdNormal::<T, 64, R>::new(T::zero(), T::one(), seed);
    let stream_seed = Cell::new(normal.stream_seed.get());
    Self {
      mu,
      sigma,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      normal,
      stream_seed,
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
      self.mu,
      self.sigma,
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
  #[inline]
  pub fn fill_slice(&self, out: &mut [T]) {
    let mm = T::splat(self.mu);
    let ss = T::splat(self.sigma);
    let mut tmp = [T::zero(); 16];
    let (chunks, rem) = out.as_chunks_mut::<16>();
    for chunk in chunks {
      self.normal.fill_16(&mut tmp);
      for half in 0..2 {
        let base = half * 8;
        let mut a = [T::zero(); 8];
        a.copy_from_slice(&tmp[base..base + 8]);
        let z = T::simd_from_array(a);
        let x = T::simd_to_array(T::simd_exp(mm + ss * z));
        chunk[base..base + 8].copy_from_slice(&x);
      }
    }
    if !rem.is_empty() {
      self.normal.fill_slice(&mut tmp[..rem.len()]);
      let mut done = 0;
      while done + 8 <= rem.len() {
        let mut a = [T::zero(); 8];
        a.copy_from_slice(&tmp[done..done + 8]);
        let z = T::simd_from_array(a);
        let x = T::simd_to_array(T::simd_exp(mm + ss * z));
        rem[done..done + 8].copy_from_slice(&x);
        done += 8;
      }
      if done < rem.len() {
        let left = rem.len() - done;
        let mut a = [T::zero(); 8];
        a[..left].copy_from_slice(&tmp[done..done + left]);
        let z = T::simd_from_array(a);
        let x = T::simd_to_array(T::simd_exp(mm + ss * z));
        rem[done..done + left].copy_from_slice(&x[..left]);
      }
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

/// LogNormal(μ=0, σ=1) — the standard log-normal, matching [`SimdNormal`]'s
/// own N(0,1) default and this file's inner `normal` sub-sampler, which is
/// always constructed as N(0,1) regardless of the caller's own μ,σ (see
/// [`SimdLogNormal::new`]).
impl<T: SimdFloatExt, R: SimdRngExt> Default for SimdLogNormal<T, R> {
  fn default() -> Self {
    Self::new(T::zero(), T::one(), &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdLogNormal<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.mu, self.sigma, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> crate::traits::DistributionExt for SimdLogNormal<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    if x <= 0.0 {
      return 0.0;
    }
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    let z = (x.ln() - mu) / sigma;
    crate::special::norm_pdf(z) / (sigma * x)
  }

  fn cdf(&self, x: f64) -> f64 {
    if x <= 0.0 {
      return 0.0;
    }
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    crate::special::norm_cdf((x.ln() - mu) / sigma)
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    (mu + sigma * crate::special::ndtri(p)).exp()
  }

  fn mean(&self) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    (mu + 0.5 * sigma * sigma).exp()
  }

  fn median(&self) -> f64 {
    self.mu.to_f64().unwrap().exp()
  }

  fn mode(&self) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    (mu - sigma * sigma).exp()
  }

  fn variance(&self) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    let s2 = sigma * sigma;
    (s2.exp() - 1.0) * (2.0 * mu + s2).exp()
  }

  fn skewness(&self) -> f64 {
    let sigma = self.sigma.to_f64().unwrap();
    let s2 = sigma * sigma;
    (s2.exp() + 2.0) * (s2.exp() - 1.0).sqrt()
  }

  fn kurtosis(&self) -> f64 {
    // Excess kurtosis.
    let sigma = self.sigma.to_f64().unwrap();
    let s2 = sigma * sigma;
    (4.0 * s2).exp() + 2.0 * (3.0 * s2).exp() + 3.0 * (2.0 * s2).exp() - 6.0
  }

  fn entropy(&self) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    0.5 + 0.5 * (2.0 * std::f64::consts::PI * sigma * sigma).ln() + mu
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdLogNormal<T, R> {
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

py_distribution!(PyLogNormal, SimdLogNormal,
  sig: (mu, sigma, seed=None, dtype=None),
  params: (mu: f64, sigma: f64)
);
