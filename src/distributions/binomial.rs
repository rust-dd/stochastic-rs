use std::cell::UnsafeCell;

use num_traits::PrimInt;
use rand::Rng;
use rand_distr::Distribution;

pub struct SimdBinomial<T: PrimInt> {
  n: u32,
  p: f64,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
}

impl<T: PrimInt> SimdBinomial<T> {
  pub fn new(n: u32, p: f64) -> Self {
    Self {
      n,
      p,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
    for x in out.iter_mut() {
      let mut count = 0u32;
      for _ in 0..self.n {
        let u: f64 = rng.random_range(0.0..1.0);
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
