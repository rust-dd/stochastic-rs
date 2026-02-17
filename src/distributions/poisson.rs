use std::cell::UnsafeCell;

use num_traits::PrimInt;
use rand::Rng;
use rand_distr::Distribution;

pub struct SimdPoisson<T: PrimInt> {
  cdf: Box<[f64]>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
}

impl<T: PrimInt> SimdPoisson<T> {
  #[inline]
  fn build_cdf(lambda: f64) -> Box<[f64]> {
    let mut cdf = Vec::new();
    let mut pmf = (-lambda).exp();
    let mut cum = pmf;
    cdf.push(cum);

    loop {
      pmf *= lambda / (cdf.len() as f64);
      cum += pmf;
      if cum >= 1.0 - 1e-15 {
        cdf.push(1.0);
        break;
      }
      cdf.push(cum);
    }

    cdf.into_boxed_slice()
  }

  pub fn new(lambda: f64) -> Self {
    assert!(lambda > 0.0);
    Self {
      cdf: Self::build_cdf(lambda),
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
    for x in out.iter_mut() {
      let u: f64 = rng.random();
      let k = self.cdf.partition_point(|&p| p < u);
      *x = num_traits::cast(k).unwrap_or(T::zero());
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

impl<T: PrimInt> Distribution<T> for SimdPoisson<T> {
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

py_distribution_int!(PyPoissonD, SimdPoisson,
  sig: (lambda_),
  params: (lambda_: f64)
);
