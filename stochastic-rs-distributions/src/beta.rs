//! # Beta
//!
//! $$
//! f(x)=\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)},\ x\in(0,1)
//! $$
//!
use std::cell::Cell;
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use super::SimdFloatExt;
use super::gamma::SimdGamma;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;

const SMALL_BETA_THRESHOLD: usize = 16;

pub struct SimdBeta<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  alpha: T,
  beta: T,
  gamma1: SimdGamma<T, R>,
  gamma2: SimdGamma<T, R>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  stream_seed: Cell<u64>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdBeta<T, R> {
  /// Creates a beta distribution with RNGs from a [`SeedExt`](crate::simd_rng::SeedExt) source.
  /// Each sub-component (gamma1, gamma2) gets an independent stream.
  pub fn new<S: crate::simd_rng::SeedExt>(alpha: T, beta: T, seed: &S) -> Self {
    assert!(alpha > T::zero() && beta > T::zero());
    let gamma1 = SimdGamma::<T, R>::new(alpha, T::one(), seed);
    let gamma2 = SimdGamma::<T, R>::new(beta, T::one(), seed);
    // No own engine to seed — reuse gamma1's already-captured stream_seed
    // as this sampler's fork anchor rather than drawing a fresh value (that
    // would shift gamma2's derivation relative to today's stream). This is
    // its own independent `Cell`, so SimdBeta's fork cursor advances on its
    // own from here on, never touching gamma1's.
    let stream_seed = Cell::new(gamma1.stream_seed.get());
    Self {
      alpha,
      beta,
      gamma1,
      gamma2,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
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
      self.alpha,
      self.beta,
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
    if out.len() < SMALL_BETA_THRESHOLD {
      for x in out.iter_mut() {
        let a = self.gamma1.sample_fast();
        let b = self.gamma2.sample_fast();
        *x = a / (a + b);
      }
      return;
    }
    let mut g1 = [T::zero(); 64];
    let mut g2 = [T::zero(); 64];
    let mut chunks = out.chunks_exact_mut(64);
    for chunk in &mut chunks {
      self.gamma1.fill_slice(&mut g1);
      self.gamma2.fill_slice(&mut g2);
      for (sub, (a8, b8)) in chunk
        .chunks_exact_mut(8)
        .zip(g1.chunks_exact(8).zip(g2.chunks_exact(8)))
      {
        let mut aa = [T::zero(); 8];
        let mut ba = [T::zero(); 8];
        aa.copy_from_slice(a8);
        ba.copy_from_slice(b8);
        let a = T::simd_from_array(aa);
        let b = T::simd_from_array(ba);
        sub.copy_from_slice(&T::simd_to_array(a / (a + b)));
      }
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      let n = rem.len();
      self.gamma1.fill_slice(&mut g1[..n]);
      self.gamma2.fill_slice(&mut g2[..n]);
      for i in 0..n {
        rem[i] = g1[i] / (g1[i] + g2[i]);
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

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdBeta<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.alpha, self.beta, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdBeta<T, R> {
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

impl<T: SimdFloatExt, R: SimdRngExt> crate::traits::DistributionExt for SimdBeta<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) {
      return 0.0;
    }
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let log_pdf = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - crate::special::ln_beta(a, b);
    log_pdf.exp()
  }

  fn cdf(&self, x: f64) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    crate::special::beta_i(a, b, x.clamp(0.0, 1.0))
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    if p <= 0.0 {
      return 0.0;
    }
    if p >= 1.0 {
      return 1.0;
    }
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    // Newton on f(x) = I_x(a,b) − p with f'(x) = pdf.
    let mut x = a / (a + b); // start at the mean
    for _ in 0..60 {
      let f = crate::special::beta_i(a, b, x) - p;
      let log_pdf = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - crate::special::ln_beta(a, b);
      let pdf = log_pdf.exp();
      if pdf <= 0.0 {
        break;
      }
      let dx = f / pdf;
      let new_x = (x - dx).clamp(1e-14, 1.0 - 1e-14);
      if (new_x - x).abs() < 1e-14 {
        return new_x;
      }
      x = new_x;
    }
    x
  }

  fn mean(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    a / (a + b)
  }

  fn median(&self) -> f64 {
    self.inv_cdf(0.5)
  }

  fn mode(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    if a > 1.0 && b > 1.0 {
      (a - 1.0) / (a + b - 2.0)
    } else {
      f64::NAN
    }
  }

  fn variance(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let s = a + b;
    a * b / (s * s * (s + 1.0))
  }

  fn skewness(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let s = a + b;
    2.0 * (b - a) * (s + 1.0).sqrt() / ((s + 2.0) * (a * b).sqrt())
  }

  fn kurtosis(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let s = a + b;
    let num = 6.0 * ((a - b).powi(2) * (s + 1.0) - a * b * (s + 2.0));
    let den = a * b * (s + 2.0) * (s + 3.0);
    num / den
  }

  fn entropy(&self) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    crate::special::ln_beta(a, b)
      - (a - 1.0) * crate::special::digamma(a)
      - (b - 1.0) * crate::special::digamma(b)
      + (a + b - 2.0) * crate::special::digamma(a + b)
  }

  fn characteristic_function(&self, _t: f64) -> num_complex::Complex64 {
    // Beta CF involves the confluent hypergeometric ₁F₁; not implemented.
    unimplemented!(
      "DistributionExt::characteristic_function for SimdBeta requires the confluent hypergeometric ₁F₁; not implemented"
    )
  }

  fn moment_generating_function(&self, _t: f64) -> f64 {
    // Closed form involves the confluent hypergeometric function 1F1.
    unimplemented!(
      "DistributionExt::moment_generating_function for SimdBeta requires the confluent hypergeometric ₁F₁; not implemented"
    )
  }
}

py_distribution!(PyBeta, SimdBeta,
  sig: (alpha, beta, seed=None, dtype=None),
  params: (alpha: f64, beta: f64)
);
