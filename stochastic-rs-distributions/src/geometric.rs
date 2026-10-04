//! # Geometric
//!
//! $$
//! \mathbb{P}(X=k)=(1-p)^{k-1}p,\ k\ge 1
//! $$
//!
//! Sampling: inversion `⌊ln U / ln(1 − p)⌋ + 1`, Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, §X.2, DOI 10.1007/978-1-4613-8643-8.

use std::marker::PhantomData;

use num_traits::PrimInt;
use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;
use wide::f64x8;

use crate::seeded::StreamState;
use crate::source::uniform53;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

const SMALL_GEOMETRIC_THRESHOLD: usize = 16;

/// Geometric law on `{1, 2, …}` with success probability `p`: parameters only; a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdGeometric<T: PrimInt> {
  p: f64,
  out: PhantomData<T>,
}

impl<T: PrimInt> SimdGeometric<T> {
  /// Creates a geometric distribution over the shifted support `k ≥ 1`
  /// (matches the module header's own convention — number of trials
  /// until, and including, the first success).
  ///
  /// - `p` — per-trial success probability p ∈ (0, 1].
  pub fn new(p: f64) -> Self {
    assert!(
      p > 0.0 && p <= 1.0,
      "p must satisfy `p > 0.0 && p <= 1.0`, got p = {p:?}"
    );
    Self {
      p,
      out: PhantomData,
    }
  }

  /// The success probability `p`.
  pub fn p(&self) -> f64 {
    self.p
  }

  /// The trial count of the floored inversion `t = ⌊ln u / ln(1 − p)⌋`, shifted onto `{1, 2, …}`; an overflow of
  /// `T` saturates (and asserts in debug).
  #[inline]
  fn count(t: f64) -> T {
    let k = t.max(0.0) + 1.0;
    let cast = num_traits::cast(k as u64);
    debug_assert!(
      cast.is_some(),
      "geometric draw {k} overflowed the output integer type"
    );
    cast.unwrap_or(T::max_value())
  }

  fn fill_parts<R: SimdRngExt>(&self, rng: &mut R, out: &mut [T]) {
    let ln1p = (1.0 - self.p).ln();
    if out.len() < SMALL_GEOMETRIC_THRESHOLD {
      let inv_ln1p = 1.0 / ln1p;
      for x in out.iter_mut() {
        *x = Self::count((rng.next_f64().ln() * inv_ln1p).floor());
      }
      return;
    }
    let inv_ln1p = f64x8::splat(1.0 / ln1p);
    let (chunks, rem) = out.as_chunks_mut::<8>();
    for chunk in chunks {
      let mut u = [0.0_f64; 8];
      rng.fill_uniform_f64(&mut u);
      let tmp = (f64x8::from(u).ln() * inv_ln1p).floor().to_array();
      for (o, &t) in chunk.iter_mut().zip(tmp.iter()) {
        *o = Self::count(t);
      }
    }
    if !rem.is_empty() {
      let mut u = [0.0_f64; 8];
      rng.fill_uniform_f64(&mut u);
      let tmp = (f64x8::from(u).ln() * inv_ln1p).floor().to_array();
      for (o, &t) in rem.iter_mut().zip(tmp.iter()) {
        *o = Self::count(t);
      }
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    let inv_ln1p = 1.0 / (1.0 - self.p).ln();
    Self::count((uniform53(rng.next_u64()).ln() * inv_ln1p).floor())
  }
}

impl<T: PrimInt + Send + Sync + 'static> Sealed for SimdGeometric<T> {}

impl<T: PrimInt + Send + Sync + 'static> SimdDistribution for SimdGeometric<T> {
  type State<R: SimdRngExt> = StreamState<T, R, 16>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (StreamState<T, R, 16>, u64) {
    StreamState::init(seed)
  }
}

impl<T: PrimInt + Send + Sync + 'static> SimdKernel for SimdGeometric<T> {
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

impl<T: PrimInt + Send + Sync + 'static> Distribution<T> for SimdGeometric<T> {
  /// The inversion of one 53-bit uniform from the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: PrimInt> crate::traits::DistributionExt for SimdGeometric<T> {
  // Convention here: support k ∈ {1, 2, ...} (the "shifted" geometric, P(X=k) = (1-p)^(k-1) p).

  fn pdf(&self, x: f64) -> Option<f64> {
    if x < 1.0 || x.fract() != 0.0 {
      return Some(0.0);
    }
    let k = x as u64;
    Some((1.0 - self.p).powi(k as i32 - 1) * self.p)
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    if x < 1.0 {
      return Some(0.0);
    }
    let k = x.floor() as u64;
    Some(1.0 - (1.0 - self.p).powi(k as i32))
  }

  fn quantile(&self, prob: f64) -> Option<f64> {
    // Smallest k such that 1-(1-p)^k ≥ prob ⟹ k = ⌈ln(1-prob)/ln(1-p)⌉
    if prob <= 0.0 {
      return Some(1.0);
    }
    if prob >= 1.0 {
      return Some(f64::INFINITY);
    }
    Some(((1.0 - prob).ln() / (1.0 - self.p).ln()).ceil())
  }

  fn mean(&self) -> Option<f64> {
    Some(1.0 / self.p)
  }

  fn median(&self) -> Option<f64> {
    Some((-(2.0_f64.ln()) / (1.0 - self.p).ln()).ceil())
  }

  fn mode(&self) -> Option<f64> {
    Some(1.0)
  }

  fn variance(&self) -> Option<f64> {
    Some((1.0 - self.p) / (self.p * self.p))
  }

  fn skewness(&self) -> Option<f64> {
    Some((2.0 - self.p) / (1.0 - self.p).sqrt())
  }

  fn kurtosis(&self) -> Option<f64> {
    Some(6.0 + self.p * self.p / (1.0 - self.p))
  }

  /// `p = 1.0` is a valid, documented parameter (see [`Self::new`]) — the
  /// degenerate distribution that always succeeds on the first trial, with
  /// zero entropy. Computed naively, `q * q.ln()` at `q = 1.0 - p = 0.0` is
  /// the indeterminate form `0.0 * (-inf) = NaN`; the standard convention
  /// `x * ln(x) -> 0` as `x -> 0+` (universal in entropy formulas, e.g.
  /// Shannon entropy) is applied explicitly instead so `p = 1.0` returns
  /// the mathematically correct `0.0` rather than `NaN`.
  fn entropy(&self) -> Option<f64> {
    let q = 1.0 - self.p;
    let q_term = if q > 0.0 { q * q.ln() } else { 0.0 };
    Some(-(q_term + self.p * self.p.ln()) / self.p)
  }

  fn characteristic_function(&self, t: f64) -> Option<num_complex::Complex64> {
    if t == 0.0 {
      return Some(num_complex::Complex64::new(1.0, 0.0));
    }
    // φ(t) = p e^{it} / (1 - (1-p) e^{it})
    let eit = num_complex::Complex64::new(0.0, t).exp();
    Some(eit.scale(self.p) / (num_complex::Complex64::new(1.0, 0.0) - eit.scale(1.0 - self.p)))
  }

  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    let q = 1.0 - self.p;
    if t == 0.0 {
      Some(1.0)
    } else if q * t.exp() < 1.0 {
      Some(self.p * t.exp() / (1.0 - q * t.exp()))
    } else {
      Some(f64::INFINITY)
    }
  }
}

py_distribution_int!(PyGeometric, SimdGeometric,
  sig: (p, seed=None),
  params: (p: f64)
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::tests::scalar_chi_square_best_p;
  use crate::traits::DistributionExt;
  use crate::traits::DistributionSampler;

  /// Backs `entropy`'s doc comment: `p = 1.0` is a valid, documented
  /// parameter (always succeeds first try), and its entropy must be the
  /// mathematically correct `0.0`, not `NaN` from a naive `0 * ln(0)`.
  #[test]
  fn entropy_at_p_one_is_zero_not_nan() {
    let g = SimdGeometric::<u64>::new(1.0);
    assert_eq!(
      g.entropy().unwrap(),
      0.0,
      "p=1.0 is degenerate, entropy must be exactly 0"
    );
  }

  /// Away from the boundary, entropy must still be a finite, positive
  /// number (uncertainty is strictly positive whenever `p < 1`).
  #[test]
  fn entropy_at_interior_p_is_finite_and_positive() {
    let g = SimdGeometric::<u64>::new(0.3);
    let h = g.entropy().unwrap();
    assert!(h.is_finite() && h > 0.0, "entropy at p=0.3 was {h}");
  }

  /// `new`'s own doc comment documents `p ∈ (0, 1]` — `p = 1.0` is the
  /// closed upper boundary, not an excluded limit, and must construct
  /// without panicking (the entropy fix above already relies on this).
  #[test]
  fn new_accepts_p_one() {
    let _ = SimdGeometric::<u64>::new(1.0);
  }

  /// `p = 0.0` sits just outside the documented domain `(0, 1]` (a
  /// per-trial success probability of zero means the waiting time is
  /// never finite) and must be rejected at construction, not silently
  /// turned into garbage output.
  #[test]
  #[should_panic(expected = "p must satisfy `p > 0.0 && p <= 1.0`")]
  fn new_rejects_p_zero() {
    let _ = SimdGeometric::<u64>::new(0.0);
  }

  /// `p > 1.0` is not a probability at all and must be rejected the same
  /// way `p = 0.0` is.
  #[test]
  #[should_panic(expected = "p must satisfy `p > 0.0 && p <= 1.0`")]
  fn new_rejects_p_above_one() {
    let _ = SimdGeometric::<u64>::new(1.5);
  }

  /// Ties `fill_slice`'s own empirical mean/variance to `mean()`/
  /// `variance()` across several `p`, including the degenerate `p = 1.0`,
  /// tolerance derived from each plug-in estimator's standard error. This
  /// is the test whose absence let the sampler (previously the
  /// `{0, 1, ...}` "failures before the first success" convention) and
  /// `pdf`/`cdf`/`mean` (the `{1, 2, ...}` "trials" convention this module
  /// documents) silently describe two different distributions: the
  /// sampler's empirical mean tracked `(1-p)/p`, not `mean()`'s `1/p`, and
  /// every draw at `p = 1.0` was `0`, never the `1` the shifted convention
  /// requires. Fails if either side reverts to the other convention.
  #[test]
  fn sampler_moments_match_analytics_across_p() {
    const N: usize = 200_000;
    for &p in &[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] {
      let dist = SimdGeometric::<u64>::new(p);
      let mut buf = vec![0u64; N];
      dist.seeded(&Deterministic::new(7)).fill_slice(&mut buf);

      if p >= 1.0 {
        let bad = buf.iter().filter(|&&x| x != 1).count();
        assert_eq!(bad, 0, "p=1.0: {bad}/{N} draws were not 1");
        assert_eq!(
          dist.mean().unwrap(),
          1.0,
          "p=1.0: mean() must be exactly 1.0"
        );
        assert_eq!(
          dist.variance().unwrap(),
          0.0,
          "p=1.0: variance() must be exactly 0.0"
        );
        continue;
      }

      let n = N as f64;
      let mean = buf.iter().map(|&x| x as f64).sum::<f64>() / n;
      let var = buf
        .iter()
        .map(|&x| {
          let d = x as f64 - mean;
          d * d
        })
        .sum::<f64>()
        / n;

      let expected_mean = dist.mean().unwrap();
      let expected_var = dist.variance().unwrap();
      // mu4 from the (shift-invariant) excess-kurtosis closed form gives
      // the plug-in variance estimator's own standard error:
      // Var(S^2) ~= (mu4 - sigma^4) / n.
      let mu4 = (dist.kurtosis().unwrap() + 3.0) * expected_var * expected_var;
      let se_mean = (expected_var / n).sqrt();
      let se_var = ((mu4 - expected_var * expected_var) / n).sqrt();

      assert!(
        (mean - expected_mean).abs() < 6.0 * se_mean,
        "p={p}: sample mean {mean} vs mean() {expected_mean} (6*SE={})",
        6.0 * se_mean
      );
      assert!(
        (var - expected_var).abs() < 6.0 * se_var,
        "p={p}: sample variance {var} vs variance() {expected_var} (6*SE={})",
        6.0 * se_var
      );
    }
  }

  /// The scalar path taken for buffers shorter than
  /// `SMALL_GEOMETRIC_THRESHOLD` shares the same `+ 1` shift as the SIMD
  /// path exercised above; a tiny buffer exercises it directly so both
  /// code paths are pinned to the same support floor.
  #[test]
  fn scalar_path_respects_shifted_support() {
    let mut dist = SimdGeometric::<u64>::new(0.4).seeded(&Deterministic::new(5));
    let mut buf = [0u64; 8];
    dist.fill_slice(&mut buf);
    assert!(buf.iter().all(|&x| x >= 1), "scalar-path draws: {buf:?}");
  }

  /// The honest `Distribution` draws from the caller's rng and agrees with the cdf.
  #[test]
  fn scalar_sample_matches_cdf() {
    let d = SimdGeometric::<u64>::new(0.15);
    let best = scalar_chi_square_best_p(&d, (1, 50), |k| d.cdf(k as f64).unwrap());
    assert!(best > 0.01, "best p = {best}");
  }
}
