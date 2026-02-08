use std::cell::UnsafeCell;

use num_traits::PrimInt;
use rand::Rng;
use rand_distr::Distribution;
use wide::f64x8;

pub struct SimdGeometric<T: PrimInt> {
  p: f64,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
}

impl<T: PrimInt> SimdGeometric<T> {
  pub fn new(p: f64) -> Self {
    Self {
      p,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
    let ln1p = (1.0 - self.p).ln();
    let inv_ln1p = f64x8::splat(1.0 / ln1p);
    let mut u = [0.0f64; 8];

    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      for v in u.iter_mut() {
        *v = rng.random_range(0.0..1.0);
      }
      let v = f64x8::from(u);
      let g = (v.ln() * inv_ln1p).floor();
      let tmp = g.to_array();
      for (o, &t) in chunk.iter_mut().zip(tmp.iter()) {
        *o = num_traits::cast(t.max(0.0) as u64).unwrap_or(T::zero());
      }
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      for v in u.iter_mut() {
        *v = rng.random_range(0.0..1.0);
      }
      let v = f64x8::from(u);
      let g = (v.ln() * inv_ln1p).floor();
      let tmp = g.to_array();
      for i in 0..rem.len() {
        rem[i] = num_traits::cast(tmp[i].max(0.0) as u64).unwrap_or(T::zero());
      }
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

impl<T: PrimInt> Distribution<T> for SimdGeometric<T> {
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
