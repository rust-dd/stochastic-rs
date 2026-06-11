//! # Poisson
//!
//! $$
//! \mathbb{P}(N=k)=e^{-\lambda}\frac{\lambda^k}{k!},\ k\in\mathbb N_0
//! $$
//!
use std::cell::UnsafeCell;

use num_traits::PrimInt;
use rand::Rng;
use rand_distr::Distribution;

use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;

pub struct SimdPoisson<T: PrimInt, R: SimdRngExt = SimdRng> {
  lambda: f64,
  cdf: Box<[f64]>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<R>,
}

impl<T: PrimInt, R: SimdRngExt> SimdPoisson<T, R> {
  /// Builds the cumulative table from log-space pmf increments
  /// `ln pmf_k = -λ + k ln λ - ln Γ(k+1)`. The naive multiplicative
  /// recurrence starts at `exp(-λ)`, which underflows to 0 for λ ≳ 745 and
  /// then never recovers — the loop would run forever. The secondary stop
  /// condition covers accumulated-rounding cases where `cum` converges a
  /// few ulp below the `1 - 1e-15` target: past `2λ` with pmf < 4e-18 the
  /// remaining tail mass is below the table's own epsilon.
  #[inline]
  fn build_cdf(lambda: f64) -> Box<[f64]> {
    let mut cdf = Vec::new();
    let ln_lambda = lambda.ln();
    let mut log_pmf = -lambda;
    let mut cum = log_pmf.exp();
    cdf.push(cum);

    loop {
      let k = cdf.len() as f64;
      log_pmf += ln_lambda - k.ln();
      cum += log_pmf.exp();
      if cum >= 1.0 - 1e-15 || (k > 2.0 * lambda && log_pmf < -40.0) {
        cdf.push(1.0);
        break;
      }
      cdf.push(cum);
    }

    cdf.into_boxed_slice()
  }

  pub fn new<S: crate::simd_rng::SeedExt>(lambda: f64, seed: &S) -> Self {
    assert!(lambda > 0.0);
    Self {
      lambda,
      cdf: Self::build_cdf(lambda),
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      simd_rng: UnsafeCell::new(seed.rng_ext::<R>()),
    }
  }

  /// Returns a single sample using the internal SIMD RNG.
  #[inline]
  pub fn sample_fast(&self) -> T {
    let index = unsafe { &mut *self.index.get() };
    if *index >= 16 {
      self.refill_buffer_fast();
    }
    let buf = unsafe { &mut *self.buffer.get() };
    let z = buf[*index];
    *index += 1;
    z
  }

  fn refill_buffer_fast(&self) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    let buf = unsafe { &mut *self.buffer.get() };
    self.fill_slice(rng, buf);
    unsafe {
      *self.index.get() = 0;
    }
  }

  /// Fills `out` using the internal SIMD RNG.
  #[inline]
  pub fn fill_slice_fast(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    self.fill_slice(rng, out);
  }

  pub fn fill_slice<Rr: Rng + ?Sized>(&self, rng: &mut Rr, out: &mut [T]) {
    for x in out.iter_mut() {
      let u: f64 = rng.random();
      let k = self.cdf.partition_point(|&p| p < u);
      *x = num_traits::cast(k).unwrap_or(T::zero());
    }
  }

  fn refill_buffer<Rr: Rng + ?Sized>(&self, rng: &mut Rr) {
    let buf = unsafe { &mut *self.buffer.get() };
    self.fill_slice(rng, buf);
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl<T: PrimInt, R: SimdRngExt> Clone for SimdPoisson<T, R> {
  fn clone(&self) -> Self {
    Self {
      lambda: self.lambda,
      cdf: self.cdf.clone(),
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      simd_rng: UnsafeCell::new(R::new()),
    }
  }
}

impl<T: PrimInt, R: SimdRngExt> Distribution<T> for SimdPoisson<T, R> {
  fn sample<Rr: Rng + ?Sized>(&self, rng: &mut Rr) -> T {
    let idx = unsafe { &mut *self.index.get() };
    if *idx >= 16 {
      self.refill_buffer(rng);
    }
    let val = unsafe { (*self.buffer.get())[*idx] };
    *idx += 1;
    val
  }
}

impl<T: PrimInt, R: SimdRngExt> SimdPoisson<T, R> {
  /// The rate parameter λ (stored at construction — the old
  /// `-ln(cdf[0])` recovery breaks down once `e^{-λ}` underflows).
  #[inline]
  fn lambda(&self) -> f64 {
    self.lambda
  }
}

impl<T: PrimInt, R: SimdRngExt> crate::traits::DistributionExt for SimdPoisson<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    if x < 0.0 || x.fract() != 0.0 {
      return 0.0;
    }
    let k = x as i64;
    let lambda = self.lambda();
    // P(N=k) = exp(−λ) λ^k / k! = exp(k ln λ − λ − ln Γ(k+1))
    let log_pmf = k as f64 * lambda.ln() - lambda - crate::special::ln_gamma((k + 1) as f64);
    log_pmf.exp()
  }

  fn cdf(&self, x: f64) -> f64 {
    if x < 0.0 {
      return 0.0;
    }
    let k = x.floor() as usize;
    if k >= self.cdf.len() {
      1.0
    } else {
      self.cdf[k]
    }
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    if p <= 0.0 {
      return 0.0;
    }
    if p >= 1.0 {
      return f64::INFINITY;
    }
    // Use the cached cumulative table built in `build_cdf`.
    match self.cdf.iter().position(|&c| c >= p) {
      Some(k) => k as f64,
      None => (self.cdf.len() - 1) as f64,
    }
  }

  fn mean(&self) -> f64 {
    self.lambda()
  }

  fn median(&self) -> f64 {
    // Approximation: ⌊λ + 1/3 - 0.02/λ⌋
    let l = self.lambda();
    (l + 1.0 / 3.0 - 0.02 / l).floor()
  }

  fn mode(&self) -> f64 {
    self.lambda().floor()
  }

  fn variance(&self) -> f64 {
    self.lambda()
  }

  fn skewness(&self) -> f64 {
    1.0 / self.lambda().sqrt()
  }

  fn kurtosis(&self) -> f64 {
    1.0 / self.lambda()
  }

  fn entropy(&self) -> f64 {
    // Closed form not elementary; fall back to an asymptotic expansion that's
    // accurate to leading order: H(λ) ≈ ½ ln(2π e λ) - 1/(12λ) - 1/(24λ²) - ...
    let l = self.lambda();
    0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * l).ln()
      - 1.0 / (12.0 * l)
      - 1.0 / (24.0 * l * l)
      - 19.0 / (360.0 * l.powi(3))
  }

  fn characteristic_function(&self, t: f64) -> num_complex::Complex64 {
    // φ(t) = exp(λ (e^{it} - 1))
    let eit = num_complex::Complex64::new(0.0, t).exp();
    (eit - num_complex::Complex64::new(1.0, 0.0))
      .scale(self.lambda())
      .exp()
  }

  fn moment_generating_function(&self, t: f64) -> f64 {
    (self.lambda() * (t.exp() - 1.0)).exp()
  }
}

py_distribution_int!(PyPoissonD, SimdPoisson,
  sig: (lambda_, seed=None),
  params: (lambda_: f64)
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::SimdPoisson;
  use crate::traits::DistributionExt;

  /// λ ≳ 745 made the old multiplicative table build spin forever on an
  /// underflowed `exp(-λ)`; the log-space build must terminate and sample
  /// with the right mean.
  #[test]
  fn poisson_large_lambda_table_terminates() {
    let dist = SimdPoisson::<u64>::new(800.0, &Deterministic::new(3));
    let mut buf = vec![0u64; 4096];
    dist.fill_slice_fast(&mut buf);
    let mean = buf.iter().map(|&x| x as f64).sum::<f64>() / buf.len() as f64;
    assert!(
      (mean - 800.0).abs() < 3.0,
      "λ=800 sample mean drift: {mean}"
    );
    assert!((dist.mean() - 800.0).abs() < 1e-9);
  }

  /// Log-space build must reproduce the small-λ table semantics.
  #[test]
  fn poisson_small_lambda_moments() {
    let dist = SimdPoisson::<u32>::new(3.5, &Deterministic::new(11));
    let mut buf = vec![0u32; 100_000];
    dist.fill_slice_fast(&mut buf);
    let n = buf.len() as f64;
    let mean = buf.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = buf
      .iter()
      .map(|&x| {
        let d = x as f64 - mean;
        d * d
      })
      .sum::<f64>()
      / n;
    assert!((mean - 3.5).abs() < 0.05, "mean drift: {mean}");
    assert!((var - 3.5).abs() < 0.15, "variance drift: {var}");
  }
}
