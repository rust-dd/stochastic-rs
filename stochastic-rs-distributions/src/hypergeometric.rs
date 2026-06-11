//! # Hypergeometric
//!
//! $$
//! \mathbb{P}(X=k)=\frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}
//! $$
//!
//! Sampling: inverse transform on a cumulative table precomputed over the
//! support via the log-space pmf recurrence — one uniform plus a binary
//! search per draw instead of the naive `n_draws` Bernoulli trials.
//!
//! Reference: Kachitvichyanukul, V., Schmeiser, B.W. (1985), "Computer
//! generation of hypergeometric random variates", *Journal of Statistical
//! Computation and Simulation* 22, 127-145, DOI: 10.1080/00949658508810839
//! (inverse-transform family).
use std::cell::UnsafeCell;

use num_traits::PrimInt;
use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;

pub struct SimdHypergeometric<T: PrimInt, R: SimdRngExt = SimdRng> {
  n_total: u32,
  k_success: u32,
  n_draws: u32,
  k_min: u32,
  cdf: Box<[f64]>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<R>,
}

impl<T: PrimInt, R: SimdRngExt> SimdHypergeometric<T, R> {
  /// Builds the cumulative table over the support `[k_min, k_max]`. The
  /// pmf recurrence runs in log space, so edge-of-support underflow (far
  /// tails of large populations) only zeroes the negligible tail terms
  /// instead of poisoning the whole table.
  fn build_cdf(n_total: u32, k_success: u32, n_draws: u32) -> (u32, Box<[f64]>) {
    let nn = n_total as f64;
    let kk = k_success as f64;
    let nd = n_draws as f64;
    let k_min = n_draws.saturating_sub(n_total - k_success);
    let k_max = n_draws.min(k_success);
    let ln_c = |a: f64, b: f64| {
      crate::special::ln_gamma(a + 1.0)
        - crate::special::ln_gamma(b + 1.0)
        - crate::special::ln_gamma(a - b + 1.0)
    };
    let mut log_pmf =
      ln_c(kk, k_min as f64) + ln_c(nn - kk, nd - k_min as f64) - ln_c(nn, nd);
    let len = (k_max - k_min + 1) as usize;
    let mut cdf = Vec::with_capacity(len);
    let mut cum = log_pmf.exp();
    cdf.push(cum.min(1.0));
    for k in k_min..k_max {
      let kf = k as f64;
      log_pmf += ((kk - kf) * (nd - kf)).ln() - ((kf + 1.0) * (nn - kk - nd + kf + 1.0)).ln();
      cum += log_pmf.exp();
      cdf.push(cum.min(1.0));
    }
    cdf[len - 1] = 1.0;
    (k_min, cdf.into_boxed_slice())
  }

  pub fn new<S: crate::simd_rng::SeedExt>(
    n_total: u32,
    k_success: u32,
    n_draws: u32,
    seed: &S,
  ) -> Self {
    assert!(k_success <= n_total, "k_success must be ≤ n_total");
    assert!(n_draws <= n_total, "n_draws must be ≤ n_total");
    let (k_min, cdf) = Self::build_cdf(n_total, k_success, n_draws);
    Self {
      n_total,
      k_success,
      n_draws,
      k_min,
      cdf,
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
      let k = self.k_min as usize + self.cdf.partition_point(|&p| p < u);
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

impl<T: PrimInt, R: SimdRngExt> Clone for SimdHypergeometric<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.n_total, self.k_success, self.n_draws, &Unseeded)
  }
}

impl<T: PrimInt, R: SimdRngExt> Distribution<T> for SimdHypergeometric<T, R> {
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

impl<T: PrimInt, R: SimdRngExt> crate::traits::DistributionExt for SimdHypergeometric<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    if x < 0.0 || x.fract() != 0.0 {
      return 0.0;
    }
    let k = x as i64;
    let big_n = self.n_total as i64;
    let big_k = self.k_success as i64;
    let n = self.n_draws as i64;
    let k_min = (n - (big_n - big_k)).max(0);
    let k_max = n.min(big_k);
    if k < k_min || k > k_max {
      return 0.0;
    }
    // P = C(K, k) C(N−K, n−k) / C(N, n) — compute via log-Γ for stability.
    let lg = |z: i64| crate::special::ln_gamma((z + 1) as f64);
    let log_pmf = lg(big_k) - lg(k) - lg(big_k - k) + lg(big_n - big_k)
      - lg(n - k)
      - lg(big_n - big_k - (n - k))
      - (lg(big_n) - lg(n) - lg(big_n - n));
    log_pmf.exp()
  }

  fn cdf(&self, x: f64) -> f64 {
    if x < 0.0 {
      return 0.0;
    }
    let k = x.floor() as i64;
    let big_n = self.n_total as i64;
    let big_k = self.k_success as i64;
    let n = self.n_draws as i64;
    let k_min = (n - (big_n - big_k)).max(0);
    let k_max = n.min(big_k);
    if k >= k_max {
      return 1.0;
    }
    // No closed form — sum the pmf from k_min to ⌊x⌋.
    let mut acc = 0.0;
    for j in k_min..=k {
      acc += self.pdf(j as f64);
    }
    acc.clamp(0.0, 1.0)
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    let big_n = self.n_total as i64;
    let big_k = self.k_success as i64;
    let n = self.n_draws as i64;
    let k_min = (n - (big_n - big_k)).max(0);
    let k_max = n.min(big_k);
    if p <= 0.0 {
      return k_min as f64;
    }
    if p >= 1.0 {
      return k_max as f64;
    }
    let mut acc = 0.0;
    for j in k_min..=k_max {
      acc += self.pdf(j as f64);
      if acc >= p {
        return j as f64;
      }
    }
    k_max as f64
  }

  fn mean(&self) -> f64 {
    self.n_draws as f64 * self.k_success as f64 / self.n_total as f64
  }

  fn median(&self) -> f64 {
    self.mean().floor()
  }

  fn mode(&self) -> f64 {
    let n = self.n_draws as f64;
    let k = self.k_success as f64;
    let big_n = self.n_total as f64;
    ((n + 1.0) * (k + 1.0) / (big_n + 2.0)).floor()
  }

  fn variance(&self) -> f64 {
    let n = self.n_draws as f64;
    let k = self.k_success as f64;
    let big_n = self.n_total as f64;
    n * k * (big_n - k) * (big_n - n) / (big_n * big_n * (big_n - 1.0))
  }

  fn skewness(&self) -> f64 {
    let n = self.n_draws as f64;
    let k = self.k_success as f64;
    let big_n = self.n_total as f64;
    ((big_n - 2.0 * k) * (big_n - 1.0).sqrt() * (big_n - 2.0 * n))
      / ((n * k * (big_n - k) * (big_n - n)).sqrt() * (big_n - 2.0))
  }

  fn moment_generating_function(&self, _t: f64) -> f64 {
    // MGF involves the hypergeometric function; not implemented in closed form.
    unimplemented!(
      "DistributionExt::moment_generating_function for SimdHypergeometric requires the Gauss hypergeometric ₂F₁; not implemented"
    )
  }
}

py_distribution_int!(PyHypergeometric, SimdHypergeometric,
  sig: (n_total, k_success, n_draws, seed=None),
  params: (n_total: u32, k_success: u32, n_draws: u32)
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::SimdHypergeometric;
  use crate::traits::DistributionExt;

  #[test]
  fn hypergeometric_inversion_matches_population_moments() {
    let dist = SimdHypergeometric::<u32>::new(500, 200, 100, &Deterministic::new(21));
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
    assert!(buf.iter().all(|&x| x <= 100));
    assert!(
      (mean - dist.mean()).abs() < 0.1,
      "mean drift: {mean} vs {}",
      dist.mean()
    );
    assert!(
      (var / dist.variance() - 1.0).abs() < 0.05,
      "variance drift: {var} vs {}",
      dist.variance()
    );
  }

  #[test]
  fn hypergeometric_pmf_matches_empirical() {
    const SAMPLES: usize = 200_000;
    let dist = SimdHypergeometric::<u32>::new(60, 25, 20, &Deterministic::new(5));
    let mut buf = vec![0u32; SAMPLES];
    dist.fill_slice_fast(&mut buf);
    let mut counts = [0usize; 21];
    for &x in &buf {
      counts[x as usize] += 1;
    }
    for (k, &got) in counts.iter().enumerate().take(14).skip(4) {
      let pmf = dist.pdf(k as f64);
      let expected = pmf * SAMPLES as f64;
      let se = (expected * (1.0 - pmf)).sqrt();
      assert!(
        (got as f64 - expected).abs() < 5.0 * se + 1.0,
        "PMF mismatch at k={k}: got {got}, expected {expected}"
      );
    }
  }
}
