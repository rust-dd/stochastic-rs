//! # Binomial
//!
//! $$
//! \mathbb{P}(X=k)=\binom{n}{k}p^k(1-p)^{n-k}
//! $$
//!
//! Sampling: BTRS transformed rejection for $n\,\min(p, 1-p) \ge 10$,
//! geometric waiting-time inversion below that threshold.
//!
//! References:
//! - Hörmann, W. (1993), "The generation of binomial random variates",
//!   *Journal of Statistical Computation and Simulation* 46, 101-110,
//!   DOI: 10.1080/00949659308811496 (Algorithm BTRS).
//! - Devroye, L. (1986), *Non-Uniform Random Variate Generation*,
//!   Springer, §X.4 (waiting-time method).
use std::marker::PhantomData;

use num_traits::PrimInt;
use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use crate::seeded::StreamState;
use crate::source::uniform53;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// Cached setup constants for Hörmann's BTRS sampler over
/// `Binomial(n, p_eff)` with `p_eff = min(p, 1-p)`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct BtrsConstants {
  flipped: bool,
  a: f64,
  b: f64,
  c: f64,
  v_r: f64,
  alpha: f64,
  lpq: f64,
  m: f64,
  h: f64,
}

impl BtrsConstants {
  /// Computes the BTRS constants when the algorithm's validity bound
  /// `n · min(p, 1−p) ≥ 10` holds; returns `None` otherwise.
  fn setup(n: u32, p: f64) -> Option<Self> {
    let flipped = p > 0.5;
    let p_eff = if flipped { 1.0 - p } else { p };
    let nf = n as f64;
    if p_eff <= 0.0 || nf * p_eff < 10.0 {
      return None;
    }
    let q = 1.0 - p_eff;
    let spq = (nf * p_eff * q).sqrt();
    let b = 1.15 + 2.53 * spq;
    let a = -0.0873 + 0.0248 * b + 0.01 * p_eff;
    let c = nf * p_eff + 0.5;
    let v_r = 0.92 - 4.2 / b;
    let alpha = (2.83 + 5.1 / b) * spq;
    let lpq = (p_eff / q).ln();
    let m = ((nf + 1.0) * p_eff).floor();
    let h = crate::special::ln_gamma(m + 1.0) + crate::special::ln_gamma(nf - m + 1.0);
    Some(Self {
      flipped,
      a,
      b,
      c,
      v_r,
      alpha,
      lpq,
      m,
      h,
    })
  }

  /// One BTRS draw of `k ~ Binomial(n, p_eff)`; the caller applies the
  /// complement flip when `p > 0.5`. Expected ~1.15 (u, v) pairs per
  /// draw; the squeeze step returns without any transcendental call
  /// ~86 % of the time.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G, n: u32) -> u32 {
    let nf = n as f64;
    loop {
      let u = uniform53(rng.next_u64()) - 0.5;
      let v = uniform53(rng.next_u64());
      let us = 0.5 - u.abs();
      let k = ((2.0 * self.a / us + self.b) * u + self.c).floor();
      if k < 0.0 || k > nf {
        continue;
      }
      if us >= 0.07 && v <= self.v_r {
        return k as u32;
      }
      let log_accept =
        self.h - crate::special::ln_gamma(k + 1.0) - crate::special::ln_gamma(nf - k + 1.0)
          + (k - self.m) * self.lpq;
      if (v * self.alpha / (self.a / (us * us) + self.b)).ln() <= log_accept {
        return k as u32;
      }
    }
  }
}

/// Binomial law of `n` trials with success probability `p`: parameters only; a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdBinomial<T: PrimInt> {
  n: u32,
  p: f64,
  btrs: Option<BtrsConstants>,
  out: PhantomData<T>,
}

impl<T: PrimInt> SimdBinomial<T> {
  /// Creates a binomial sampler.
  ///
  /// - `n` — number of Bernoulli trials (matches the module header's n).
  /// - `p` — per-trial success probability p ∈ [0, 1] (matches the
  ///   module header's p).
  pub fn new(n: u32, p: f64) -> Self {
    assert!(
      (0.0..=1.0).contains(&p),
      "p must satisfy `0.0 <= p <= 1.0`, got p = {p:?}"
    );
    Self {
      n,
      p,
      btrs: BtrsConstants::setup(n, p),
      out: PhantomData,
    }
  }

  /// The number of trials `n`.
  pub fn n(&self) -> u32 {
    self.n
  }

  /// The success probability `p`.
  pub fn p(&self) -> f64 {
    self.p
  }

  /// `k` as `T`; an overflow saturates to `T::max_value()` (and asserts in debug) rather than reporting `0`.
  #[inline]
  fn cast(k: u32) -> T {
    let cast = num_traits::cast(k);
    debug_assert!(
      cast.is_some(),
      "binomial draw {k} overflowed the output integer type"
    );
    cast.unwrap_or(T::max_value())
  }

  fn fill_parts<G: Rng + ?Sized>(&self, rng: &mut G, out: &mut [T]) {
    if self.n == 0 {
      for x in out.iter_mut() {
        *x = T::zero();
      }
      return;
    }

    let flipped = self.p > 0.5;
    let p_eff = if flipped { 1.0 - self.p } else { self.p };

    if p_eff <= 0.0 {
      let cast = Self::cast(if flipped { self.n } else { 0 });
      for x in out.iter_mut() {
        *x = cast;
      }
      return;
    }

    if let Some(btrs) = &self.btrs {
      for x in out.iter_mut() {
        let k = btrs.sample(rng, self.n);
        *x = Self::cast(if btrs.flipped { self.n - k } else { k });
      }
      return;
    }

    let log_q = (1.0 - p_eff).ln();
    for x in out.iter_mut() {
      let mut count: u32 = 0;
      let mut total: i64 = 0;
      loop {
        let u = uniform53(rng.next_u64());
        let g = (u.ln() / log_q).floor() as i64 + 1;
        total += g;
        if total > self.n as i64 {
          break;
        }
        count += 1;
      }
      *x = Self::cast(if flipped { self.n - count } else { count });
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    let mut out = [T::zero()];
    self.fill_parts(rng, &mut out);
    out[0]
  }
}

impl<T: PrimInt + Send + Sync + 'static> Sealed for SimdBinomial<T> {}

impl<T: PrimInt + Send + Sync + 'static> SimdDistribution for SimdBinomial<T> {
  type State<R: SimdRngExt> = StreamState<T, R, 16>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (StreamState<T, R, 16>, u64) {
    StreamState::init(seed)
  }
}

impl<T: PrimInt + Send + Sync + 'static> SimdKernel for SimdBinomial<T> {
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

impl<T: PrimInt + Send + Sync + 'static> Distribution<T> for SimdBinomial<T> {
  /// One BTRS or waiting-time draw on the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
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
  use crate::tests::scalar_chi_square_best_p;
  use crate::traits::DistributionExt;
  use crate::traits::DistributionSampler;

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
    let dist = SimdBinomial::<u32>::new(n, p);
    let mut buf = vec![0u32; 50_000];
    dist.seeded(&Deterministic::new(42)).fill_slice(&mut buf);
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
    let dist = SimdBinomial::<u32>::new(n, p);
    let mut buf = vec![0u32; 50_000];
    dist.seeded(&Deterministic::new(7)).fill_slice(&mut buf);
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
    let dist = SimdBinomial::<u32>::new(n, p);
    let mut buf = vec![0u32; 20_000];
    dist.seeded(&Deterministic::new(11)).fill_slice(&mut buf);
    let (mean, var) = moments(&buf);
    assert!(buf.iter().all(|&x| x <= n));
    assert!((mean - dist.mean()).abs() < 0.5);
    assert!((var / dist.variance() - 1.0).abs() < 0.15);
  }

  #[test]
  fn n_zero_returns_zeros() {
    let dist = SimdBinomial::<u32>::new(0, 0.5);
    let mut buf = vec![0u32; 32];
    dist.seeded(&Unseeded).fill_slice(&mut buf);
    assert!(buf.iter().all(|&x| x == 0));
  }

  #[test]
  fn btrs_path_matches_population_moments() {
    let n = 1_000u32;
    let p = 0.37;
    let dist = SimdBinomial::<u32>::new(n, p);
    let mut buf = vec![0u32; 100_000];
    dist.seeded(&Deterministic::new(1234)).fill_slice(&mut buf);
    let (mean, var) = moments(&buf);
    assert!(buf.iter().all(|&x| x <= n));
    assert!(
      (mean - dist.mean()).abs() < 0.5,
      "BTRS mean drift: got {mean}, expected {}",
      dist.mean()
    );
    assert!(
      (var / dist.variance() - 1.0).abs() < 0.05,
      "BTRS variance drift: got {var}, expected {}",
      dist.variance()
    );
  }

  #[test]
  fn btrs_pmf_matches_empirical_frequencies() {
    const SAMPLES: usize = 200_000;
    let n = 80u32;
    let p = 0.5;
    let dist = SimdBinomial::<u32>::new(n, p);
    let mut buf = vec![0u32; SAMPLES];
    dist.seeded(&Deterministic::new(99)).fill_slice(&mut buf);
    let mut counts = [0usize; 81];
    for &x in &buf {
      counts[x as usize] += 1;
    }
    for (k, _) in counts.iter().enumerate().take(50 + 1).skip(30) {
      let pmf = dist.pdf(k as f64);
      let expected = pmf * SAMPLES as f64;
      let got = counts[k] as f64;
      let se = (expected * (1.0 - pmf)).sqrt();
      assert!(
        (got - expected).abs() < 5.0 * se + 1.0,
        "PMF mismatch at k={k}: got {got}, expected {expected}"
      );
    }
  }

  /// The honest `Distribution` draws from the caller's rng and agrees with the cdf on both sampling branches.
  #[test]
  fn scalar_sample_matches_cdf() {
    for (n, p, window) in [(60, 0.4, (0, 54)), (15, 0.3, (0, 15))] {
      let d = SimdBinomial::<u32>::new(n, p);
      let best = scalar_chi_square_best_p(&d, window, |k| d.cdf(k as f64));
      assert!(best > 0.01, "Binomial({n}, {p}): best p = {best}");
    }
  }
}
