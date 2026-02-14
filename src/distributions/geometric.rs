use std::cell::UnsafeCell;

use num_traits::PrimInt;
use rand::Rng;
use rand_distr::Distribution;
use wide::f64x8;

use crate::simd_rng::SimdRng;

pub struct SimdGeometric<T: PrimInt> {
  p: f64,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<SimdRng>,
}

impl<T: PrimInt> SimdGeometric<T> {
  pub fn new(p: f64) -> Self {
    Self {
      p,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      simd_rng: UnsafeCell::new(SimdRng::new()),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, _rng: &mut R, out: &mut [T]) {
    self.fill_slice_fast(out);
  }

  pub fn fill_slice_fast(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    let ln1p = (1.0 - self.p).ln();
    let inv_ln1p = f64x8::splat(1.0 / ln1p);
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      let u = rng.next_f64_array();
      let v = f64x8::from(u);
      let g = (v.ln() * inv_ln1p).floor();
      let tmp = g.to_array();
      for (o, &t) in chunk.iter_mut().zip(tmp.iter()) {
        *o = num_traits::cast(t.max(0.0) as u64).unwrap_or(T::zero());
      }
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      let u = rng.next_f64_array();
      let v = f64x8::from(u);
      let g = (v.ln() * inv_ln1p).floor();
      let tmp = g.to_array();
      for i in 0..rem.len() {
        rem[i] = num_traits::cast(tmp[i].max(0.0) as u64).unwrap_or(T::zero());
      }
    }
  }

  fn refill_buffer(&self) {
    let buf = unsafe { &mut *self.buffer.get() };
    self.fill_slice_fast(buf);
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl<T: PrimInt> Distribution<T> for SimdGeometric<T> {
  fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> T {
    let idx = unsafe { &mut *self.index.get() };
    if *idx >= 16 {
      self.refill_buffer();
    }
    let val = unsafe { (*self.buffer.get())[*idx] };
    *idx += 1;
    val
  }
}

py_distribution_int!(PyGeometric, SimdGeometric,
  sig: (p),
  params: (p: f64)
);
