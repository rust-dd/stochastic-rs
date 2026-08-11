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
        let g = (u.ln() * inv_ln1p).floor();
        let cast = num_traits::cast(g.max(0.0) as u64);
        debug_assert!(
          cast.is_some(),
          "geometric draw {g} overflowed the output integer type"
        );
        *x = cast.unwrap_or(T::max_value());
      }
      return;
    }
    let inv_ln1p = f64x8::splat(1.0 / ln1p);
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      let mut u = [0.0_f64; 8];
      rng.fill_uniform_f64(&mut u);
      let v = f64x8::from(u);
      let tmp = (v.ln() * inv_ln1p).floor().to_array();
      for (o, &t) in chunk.iter_mut().zip(tmp.iter()) {
        let cast = num_traits::cast(t.max(0.0) as u64);
        debug_assert!(
          cast.is_some(),
          "geometric draw {t} overflowed the output integer type"
        );
        *o = cast.unwrap_or(T::max_value());
      }
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      let mut u = [0.0_f64; 8];
      rng.fill_uniform_f64(&mut u);
      let v = f64x8::from(u);
      let tmp = (v.ln() * inv_ln1p).floor().to_array();
      for i in 0..rem.len() {
        let cast = num_traits::cast(tmp[i].max(0.0) as u64);
        debug_assert!(
          cast.is_some(),
          "geometric draw {} overflowed the output integer type",
          tmp[i]
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

  fn entropy(&self) -> f64 {
    let q = 1.0 - self.p;
    -(q * q.ln() + self.p * self.p.ln()) / self.p
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
