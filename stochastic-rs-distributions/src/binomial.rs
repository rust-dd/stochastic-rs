//! # Binomial
//!
//! $$
//! \mathbb{P}(X=k)=\binom{n}{k}p^k(1-p)^{n-k}
//! $$
//!
use std::cell::UnsafeCell;

use num_traits::PrimInt;
use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::simd_rng::SimdRng;

pub struct SimdBinomial<T: PrimInt> {
  n: u32,
  p: f64,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<SimdRng>,
}

impl<T: PrimInt> SimdBinomial<T> {
  /// Creates a binomial sampler with an RNG obtained from a
  /// [`SeedExt`](crate::simd_rng::SeedExt) source. The core constructor —
  /// `new()` and `with_seed()` delegate here.
  pub fn new<S: crate::simd_rng::SeedExt>(n: u32, p: f64, seed: &S) -> Self {
    Self {
      n,
      p,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      simd_rng: UnsafeCell::new(seed.rng()),
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

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
    if self.n == 0 {
      for x in out.iter_mut() {
        *x = T::zero();
      }
      return;
    }

    let p_eff = if self.p > 0.5 { 1.0 - self.p } else { self.p };
    let np_eff = (self.n as f64) * p_eff;

    if np_eff < 30.0 && p_eff > 0.0 && p_eff < 1.0 {
      let log_q = (1.0 - p_eff).ln();
      for x in out.iter_mut() {
        let mut count: u32 = 0;
        let mut total: i64 = 0;
        loop {
          let u: f64 = rng.random();
          let g = (u.ln() / log_q).floor() as i64 + 1;
          total += g;
          if total > self.n as i64 {
            break;
          }
          count += 1;
        }
        let result = if self.p > 0.5 { self.n - count } else { count };
        *x = num_traits::cast(result).unwrap_or(T::zero());
      }
      return;
    }

    for x in out.iter_mut() {
      let mut count = 0u32;
      for _ in 0..self.n {
        let u: f64 = rng.random();
        if u < self.p {
          count += 1;
        }
      }
      *x = num_traits::cast(count).unwrap_or(T::zero());
    }
  }

  fn refill_buffer<R: Rng + ?Sized>(&self, rng: &mut R) {
    let buf = unsafe { &mut *self.buffer.get() };
    self.fill_slice(rng, buf);
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl<T: PrimInt> Clone for SimdBinomial<T> {
  fn clone(&self) -> Self {
    Self::new(self.n, self.p, &Unseeded)
  }
}

impl<T: PrimInt> Distribution<T> for SimdBinomial<T> {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
    let idx = unsafe { &mut *self.index.get() };
    if *idx >= 16 {
      self.refill_buffer(rng);
    }
    let val = unsafe { (*self.buffer.get())[*idx] };
    *idx += 1;
    val
  }
}

impl<T: PrimInt> crate::traits::DistributionExt for SimdBinomial<T> {
  fn pdf(&self, x: f64) -> f64 {
    if x < 0.0 || x.fract() != 0.0 {
      return 0.0;
    }
    let k = x as i64;
    let n = self.n as i64;
    if k > n {
      return 0.0;
    }
    // P(X=k) = exp( ln Γ(n+1) − ln Γ(k+1) − ln Γ(n−k+1) + k ln p + (n−k) ln(1−p) )
    let log_pmf = crate::special::ln_gamma((n + 1) as f64)
      - crate::special::ln_gamma((k + 1) as f64)
      - crate::special::ln_gamma((n - k + 1) as f64)
      + k as f64 * self.p.ln()
      + (n - k) as f64 * (1.0 - self.p).ln();
    log_pmf.exp()
  }

  fn cdf(&self, x: f64) -> f64 {
    if x < 0.0 {
      return 0.0;
    }
    let k = x.floor() as i64;
    let n = self.n as i64;
    if k >= n {
      return 1.0;
    }
    // CDF(k) = I_{1−p}(n−k, k+1)
    crate::special::beta_i((n - k) as f64, (k + 1) as f64, 1.0 - self.p)
  }

  fn inv_cdf(&self, prob: f64) -> f64 {
    if prob <= 0.0 {
      return 0.0;
    }
    if prob >= 1.0 {
      return self.n as f64;
    }
    // Linear scan from a Gaussian-approximation seed.
    let mean = self.n as f64 * self.p;
    let sd = (self.n as f64 * self.p * (1.0 - self.p)).sqrt();
    let mut k = (mean + crate::special::ndtri(prob) * sd)
      .round()
      .clamp(0.0, self.n as f64) as i64;
    // Walk up if cdf(k) < prob, walk down if cdf(k−1) ≥ prob.
    while self.cdf(k as f64) < prob && k < self.n as i64 {
      k += 1;
    }
    while k > 0 && self.cdf((k - 1) as f64) >= prob {
      k -= 1;
    }
    k as f64
  }

  fn mean(&self) -> f64 {
    self.n as f64 * self.p
  }

  fn median(&self) -> f64 {
    // Either floor(np) or ceil(np); pick the integer-valued median.
    (self.n as f64 * self.p).floor()
  }

  fn mode(&self) -> f64 {
    ((self.n as f64 + 1.0) * self.p).floor()
  }

  fn variance(&self) -> f64 {
    self.n as f64 * self.p * (1.0 - self.p)
  }

  fn skewness(&self) -> f64 {
    let q = 1.0 - self.p;
    (q - self.p) / (self.n as f64 * self.p * q).sqrt()
  }

  fn kurtosis(&self) -> f64 {
    let q = 1.0 - self.p;
    (1.0 - 6.0 * self.p * q) / (self.n as f64 * self.p * q)
  }

  fn characteristic_function(&self, t: f64) -> num_complex::Complex64 {
    // φ(t) = (1 - p + p e^{it})^n
    let z = num_complex::Complex64::new(1.0 - self.p, 0.0)
      + num_complex::Complex64::new(0.0, t).exp().scale(self.p);
    z.powi(self.n as i32)
  }

  fn moment_generating_function(&self, t: f64) -> f64 {
    (1.0 - self.p + self.p * t.exp()).powi(self.n as i32)
  }
}

py_distribution_int!(PyBinomial, SimdBinomial,
  sig: (n, p, seed=None),
  params: (n: u32, p: f64)
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::*;
  use crate::traits::DistributionExt;

  fn moments(samples: &[u32]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = samples
      .iter()
      .map(|&x| {
        let d = x as f64 - mean;
        d * d
      })
      .sum::<f64>()
      / n;
    (mean, var)
  }

  #[test]
  fn small_n_path_matches_population_moments() {
    let n = 20u32;
    let p = 0.3;
    let dist = SimdBinomial::<u32>::new(n, p, &Deterministic::new(42));
    let mut buf = vec![0u32; 50_000];
    dist.fill_slice_fast(&mut buf);
    let (mean, var) = moments(&buf);
    let expected_mean = dist.mean();
    let expected_var = dist.variance();
    assert!(
      (mean - expected_mean).abs() < 0.05,
      "mean drift: got {mean}, expected {expected_mean}"
    );
    assert!(
      (var - expected_var).abs() < 0.2,
      "variance drift: got {var}, expected {expected_var}"
    );
  }

  #[test]
  fn large_n_small_p_wait_path_matches_population() {
    let n = 5_000u32;
    let p = 0.005;
    let dist = SimdBinomial::<u32>::new(n, p, &Deterministic::new(7));
    let mut buf = vec![0u32; 50_000];
    dist.fill_slice_fast(&mut buf);
    let (mean, var) = moments(&buf);
    let expected_mean = dist.mean();
    let expected_var = dist.variance();
    assert!(
      (mean - expected_mean).abs() < 0.5,
      "wait-method mean drift: got {mean}, expected {expected_mean}"
    );
    assert!(
      (var / expected_var - 1.0).abs() < 0.1,
      "wait-method variance drift: got {var}, expected {expected_var}"
    );
  }

  #[test]
  fn large_n_p_close_to_one_uses_complement() {
    let n = 5_000u32;
    let p = 0.998;
    let dist = SimdBinomial::<u32>::new(n, p, &Deterministic::new(11));
    let mut buf = vec![0u32; 20_000];
    dist.fill_slice_fast(&mut buf);
    let (mean, var) = moments(&buf);
    assert!(buf.iter().all(|&x| x <= n));
    assert!((mean - dist.mean()).abs() < 0.5);
    assert!((var / dist.variance() - 1.0).abs() < 0.15);
  }

  #[test]
  fn n_zero_returns_zeros() {
    let dist = SimdBinomial::<u32>::new(0, 0.5, &Unseeded);
    let mut buf = vec![0u32; 32];
    dist.fill_slice_fast(&mut buf);
    assert!(buf.iter().all(|&x| x == 0));
  }
}
