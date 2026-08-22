//! # Geometric
//!
//! $$
//! \mathbb{P}(X=k)=(1-p)^{k-1}p,\ k\ge 1
//! $$
//!
use std::cell::Cell;
use std::cell::UnsafeCell;

use num_traits::PrimInt;
use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;
use wide::f64x8;

use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;

const SMALL_GEOMETRIC_THRESHOLD: usize = 16;

pub struct SimdGeometric<T: PrimInt, R: SimdRngExt = SimdRng> {
  p: f64,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<R>,
  stream_seed: Cell<u64>,
}

impl<T: PrimInt, R: SimdRngExt> SimdGeometric<T, R> {
  /// Creates a geometric distribution over the shifted support `k ≥ 1`
  /// (matches the module header's own convention — number of trials
  /// until, and including, the first success).
  ///
  /// - `p` — per-trial success probability p ∈ (0, 1].
  pub fn new<S: crate::simd_rng::SeedExt>(p: f64, seed: &S) -> Self {
    assert!(p > 0.0 && p <= 1.0, "p must be in (0, 1]");
    let stream_seed = seed.seed_value();
    Self {
      p,
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
    Self::new(self.p, &crate::simd_rng::Deterministic::new(child_seed))
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
  ///
  /// `floor(ln(U) / ln(1-p))` is the textbook inversion for a geometric
  /// variate on `{0, 1, 2, ...}` ("failures before the first success").
  /// This type's `pdf`/`cdf`/`mean`/`inv_cdf` below all describe the
  /// shifted `{1, 2, ...}` convention instead ("trials up to and including
  /// the first success", matching `scipy.stats.geom`), so the draw is
  /// shifted by `+ 1` onto that same support — the same shift
  /// `SimdBinomial`'s own geometric waiting-time loop applies to its
  /// inner gaps.
  ///
  /// A draw that overflows `T` saturates to `T::max_value()` rather than
  /// silently reporting a `0` count; `debug_assert!` surfaces the overflow
  /// in debug builds so undersized output types are caught during testing.
  pub fn fill_slice(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    let ln1p = (1.0 - self.p).ln();
    if out.len() < SMALL_GEOMETRIC_THRESHOLD {
      let inv_ln1p = 1.0 / ln1p;
      for x in out.iter_mut() {
        let u = rng.next_f64();
        let k = (u.ln() * inv_ln1p).floor().max(0.0) + 1.0;
        let cast = num_traits::cast(k as u64);
        debug_assert!(
          cast.is_some(),
          "geometric draw {k} overflowed the output integer type"
        );
        *x = cast.unwrap_or(T::max_value());
      }
      return;
    }
    let inv_ln1p = f64x8::splat(1.0 / ln1p);
    let (chunks, rem) = out.as_chunks_mut::<8>();
    for chunk in chunks {
      let mut u = [0.0_f64; 8];
      rng.fill_uniform_f64(&mut u);
      let v = f64x8::from(u);
      let tmp = (v.ln() * inv_ln1p).floor().to_array();
      for (o, &t) in chunk.iter_mut().zip(tmp.iter()) {
        let k = t.max(0.0) + 1.0;
        let cast = num_traits::cast(k as u64);
        debug_assert!(
          cast.is_some(),
          "geometric draw {k} overflowed the output integer type"
        );
        *o = cast.unwrap_or(T::max_value());
      }
    }
    if !rem.is_empty() {
      let mut u = [0.0_f64; 8];
      rng.fill_uniform_f64(&mut u);
      let v = f64x8::from(u);
      let tmp = (v.ln() * inv_ln1p).floor().to_array();
      for i in 0..rem.len() {
        let k = tmp[i].max(0.0) + 1.0;
        let cast = num_traits::cast(k as u64);
        debug_assert!(
          cast.is_some(),
          "geometric draw {k} overflowed the output integer type"
        );
        rem[i] = cast.unwrap_or(T::max_value());
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

impl<T: PrimInt, R: SimdRngExt> Clone for SimdGeometric<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.p, &Unseeded)
  }
}

impl<T: PrimInt, R: SimdRngExt> Distribution<T> for SimdGeometric<T, R> {
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

impl<T: PrimInt, R: SimdRngExt> crate::traits::DistributionExt for SimdGeometric<T, R> {
  // Convention here: support k ∈ {1, 2, ...} (the "shifted" geometric, P(X=k) = (1-p)^(k-1) p).

  fn pdf(&self, x: f64) -> f64 {
    if x < 1.0 || x.fract() != 0.0 {
      return 0.0;
    }
    let k = x as u64;
    (1.0 - self.p).powi(k as i32 - 1) * self.p
  }

  fn cdf(&self, x: f64) -> f64 {
    if x < 1.0 {
      return 0.0;
    }
    let k = x.floor() as u64;
    1.0 - (1.0 - self.p).powi(k as i32)
  }

  fn inv_cdf(&self, prob: f64) -> f64 {
    // Smallest k such that 1-(1-p)^k ≥ prob ⟹ k = ⌈ln(1-prob)/ln(1-p)⌉
    if prob <= 0.0 {
      return 1.0;
    }
    if prob >= 1.0 {
      return f64::INFINITY;
    }
    ((1.0 - prob).ln() / (1.0 - self.p).ln()).ceil()
  }

  fn mean(&self) -> f64 {
    1.0 / self.p
  }

  fn median(&self) -> f64 {
    (-(2.0_f64.ln()) / (1.0 - self.p).ln()).ceil()
  }

  fn mode(&self) -> f64 {
    1.0
  }

  fn variance(&self) -> f64 {
    (1.0 - self.p) / (self.p * self.p)
  }

  fn skewness(&self) -> f64 {
    (2.0 - self.p) / (1.0 - self.p).sqrt()
  }

  fn kurtosis(&self) -> f64 {
    6.0 + self.p * self.p / (1.0 - self.p)
  }

  /// `p = 1.0` is a valid, documented parameter (see [`Self::new`]) — the
  /// degenerate distribution that always succeeds on the first trial, with
  /// zero entropy. Computed naively, `q * q.ln()` at `q = 1.0 - p = 0.0` is
  /// the indeterminate form `0.0 * (-inf) = NaN`; the standard convention
  /// `x * ln(x) -> 0` as `x -> 0+` (universal in entropy formulas, e.g.
  /// Shannon entropy) is applied explicitly instead so `p = 1.0` returns
  /// the mathematically correct `0.0` rather than `NaN`.
  fn entropy(&self) -> f64 {
    let q = 1.0 - self.p;
    let q_term = if q > 0.0 { q * q.ln() } else { 0.0 };
    -(q_term + self.p * self.p.ln()) / self.p
  }

  fn characteristic_function(&self, t: f64) -> num_complex::Complex64 {
    // φ(t) = p e^{it} / (1 - (1-p) e^{it})
    let eit = num_complex::Complex64::new(0.0, t).exp();
    eit.scale(self.p) / (num_complex::Complex64::new(1.0, 0.0) - eit.scale(1.0 - self.p))
  }

  fn moment_generating_function(&self, t: f64) -> f64 {
    let q = 1.0 - self.p;
    if q * t.exp() < 1.0 {
      self.p * t.exp() / (1.0 - q * t.exp())
    } else {
      f64::INFINITY
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
  use crate::traits::DistributionExt;

  /// Backs `entropy`'s doc comment: `p = 1.0` is a valid, documented
  /// parameter (always succeeds first try), and its entropy must be the
  /// mathematically correct `0.0`, not `NaN` from a naive `0 * ln(0)`.
  #[test]
  fn entropy_at_p_one_is_zero_not_nan() {
    let g = SimdGeometric::<u64>::new(1.0, &Unseeded);
    assert_eq!(
      g.entropy(),
      0.0,
      "p=1.0 is degenerate, entropy must be exactly 0"
    );
  }

  /// Away from the boundary, entropy must still be a finite, positive
  /// number (uncertainty is strictly positive whenever `p < 1`).
  #[test]
  fn entropy_at_interior_p_is_finite_and_positive() {
    let g = SimdGeometric::<u64>::new(0.3, &Unseeded);
    let h = g.entropy();
    assert!(h.is_finite() && h > 0.0, "entropy at p=0.3 was {h}");
  }

  /// `new`'s own doc comment documents `p ∈ (0, 1]` — `p = 1.0` is the
  /// closed upper boundary, not an excluded limit, and must construct
  /// without panicking (the entropy fix above already relies on this).
  #[test]
  fn new_accepts_p_one() {
    let _ = SimdGeometric::<u64>::new(1.0, &Unseeded);
  }

  /// `p = 0.0` sits just outside the documented domain `(0, 1]` (a
  /// per-trial success probability of zero means the waiting time is
  /// never finite) and must be rejected at construction, not silently
  /// turned into garbage output.
  #[test]
  #[should_panic(expected = "p must be in (0, 1]")]
  fn new_rejects_p_zero() {
    let _ = SimdGeometric::<u64>::new(0.0, &Unseeded);
  }

  /// `p > 1.0` is not a probability at all and must be rejected the same
  /// way `p = 0.0` is.
  #[test]
  #[should_panic(expected = "p must be in (0, 1]")]
  fn new_rejects_p_above_one() {
    let _ = SimdGeometric::<u64>::new(1.5, &Unseeded);
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
      let dist = SimdGeometric::<u64>::new(p, &Deterministic::new(7));
      let mut buf = vec![0u64; N];
      dist.fill_slice(&mut buf);

      if p >= 1.0 {
        let bad = buf.iter().filter(|&&x| x != 1).count();
        assert_eq!(bad, 0, "p=1.0: {bad}/{N} draws were not 1");
        assert_eq!(dist.mean(), 1.0, "p=1.0: mean() must be exactly 1.0");
        assert_eq!(
          dist.variance(),
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

      let expected_mean = dist.mean();
      let expected_var = dist.variance();
      // mu4 from the (shift-invariant) excess-kurtosis closed form gives
      // the plug-in variance estimator's own standard error:
      // Var(S^2) ~= (mu4 - sigma^4) / n.
      let mu4 = (dist.kurtosis() + 3.0) * expected_var * expected_var;
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
    let dist = SimdGeometric::<u64>::new(0.4, &Deterministic::new(5));
    let mut buf = [0u64; 8];
    dist.fill_slice(&mut buf);
    assert!(buf.iter().all(|&x| x >= 1), "scalar-path draws: {buf:?}");
  }
}
