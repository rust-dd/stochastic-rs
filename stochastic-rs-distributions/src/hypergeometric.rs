//! # Hypergeometric
//!
//! $$
//! \mathbb{P}(X=k)=\frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}
//! $$
//!
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use stochastic_rs_core::simd_rng::Unseeded;

use num_traits::PrimInt;
use rand::Rng;
use rand_distr::Distribution;

use crate::simd_rng::SimdRng;

pub struct SimdHypergeometric<T: PrimInt> {
  n_total: u32,
  k_success: u32,
  n_draws: u32,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<SimdRng>,
  _marker: PhantomData<T>,
}

impl<T: PrimInt> SimdHypergeometric<T> {

  pub fn new<S: crate::simd_rng::SeedExt>(
    n_total: u32,
    k_success: u32,
    n_draws: u32,
    seed: &S,
  ) -> Self {
    Self {
      n_total,
      k_success,
      n_draws,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      simd_rng: UnsafeCell::new(seed.rng()),
      _marker: PhantomData,
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
    for x in out.iter_mut() {
      let mut count = 0u32;
      let mut rem_succ = self.k_success;
      let mut rem_tot = self.n_total;
      let mut draws = self.n_draws;
      while draws > 0 {
        let u: f64 = rng.random();
        if u < (rem_succ as f64) / (rem_tot as f64) {
          count += 1;
          rem_succ -= 1;
        }
        rem_tot -= 1;
        draws -= 1;
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

impl<T: PrimInt> Clone for SimdHypergeometric<T> {
  fn clone(&self) -> Self {
    Self::new(self.n_total, self.k_success, self.n_draws, &Unseeded)
  }
}

impl<T: PrimInt> Distribution<T> for SimdHypergeometric<T> {
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

impl<T: PrimInt> crate::traits::DistributionExt for SimdHypergeometric<T> {
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
