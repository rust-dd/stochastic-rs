use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::gamma::SimdGamma;
use super::SimdFloat;

pub struct SimdBeta<T: SimdFloat> {
  alpha: T,
  beta: T,
  gamma1: SimdGamma<T>,
  gamma2: SimdGamma<T>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
}

impl<T: SimdFloat> SimdBeta<T> {
  pub fn new(alpha: T, beta: T) -> Self {
    assert!(alpha > T::zero() && beta > T::zero());
    Self {
      alpha,
      beta,
      gamma1: SimdGamma::new(alpha, T::one()),
      gamma2: SimdGamma::new(beta, T::one()),
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
    let mut y1 = vec![T::zero(); out.len()];
    let mut y2 = vec![T::zero(); out.len()];
    self.gamma1.fill_slice(rng, &mut y1);
    self.gamma2.fill_slice(rng, &mut y2);
    for (o, (a, b)) in out.iter_mut().zip(y1.iter().zip(y2.iter())) {
      *o = *a / (*a + *b);
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

impl<T: SimdFloat> Distribution<T> for SimdBeta<T> {
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
