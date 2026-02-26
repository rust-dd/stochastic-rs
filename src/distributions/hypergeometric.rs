//! # Hypergeometric
//!
//! $$
//! \mathbb{P}(X=k)=\frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}
//! $$
//!
use std::cell::UnsafeCell;
use std::marker::PhantomData;

use num_traits::PrimInt;
use rand::Rng;
use rand_distr::Distribution;

pub struct SimdHypergeometric<T: PrimInt> {
  n_total: u32,
  k_success: u32,
  n_draws: u32,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  _marker: PhantomData<T>,
}

impl<T: PrimInt> SimdHypergeometric<T> {
  pub fn new(n_total: u32, k_success: u32, n_draws: u32) -> Self {
    Self {
      n_total,
      k_success,
      n_draws,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      _marker: PhantomData,
    }
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
    Self::new(self.n_total, self.k_success, self.n_draws)
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

py_distribution_int!(PyHypergeometric, SimdHypergeometric,
  sig: (n_total, k_success, n_draws),
  params: (n_total: u32, k_success: u32, n_draws: u32)
);
