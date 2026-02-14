use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::normal::SimdNormal;
use super::SimdFloatExt;
use crate::simd_rng::SimdRng;

pub struct SimdGamma<T: SimdFloatExt> {
  alpha: T,
  scale: T,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  normal: SimdNormal<T>,
  simd_rng: UnsafeCell<SimdRng>,
}

impl<T: SimdFloatExt> SimdGamma<T> {
  pub fn new(alpha: T, scale: T) -> Self {
    assert!(alpha > T::zero() && scale > T::zero());
    Self {
      alpha,
      scale,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      normal: SimdNormal::new(T::zero(), T::one()),
      simd_rng: UnsafeCell::new(SimdRng::new()),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, _rng: &mut R, out: &mut [T]) {
    self.fill_slice_fast(out);
  }

  pub fn fill_slice_fast(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    if self.alpha < T::one() {
      let gamma_plus_one = SimdGamma::new(self.alpha + T::one(), self.scale);
      for x in out.iter_mut() {
        let g = gamma_plus_one.sample(rng);
        let u: T = T::sample_uniform(rng);
        *x = g * u.powf(T::one() / self.alpha);
      }
    } else {
      let third = T::from(1.0 / 3.0).unwrap();
      let d = self.alpha - third;
      let c = T::one() / (T::from(9.0).unwrap() * d).sqrt();
      for x in out.iter_mut() {
        let val = loop {
          let z: T = self.normal.sample(rng);
          let v = (T::one() + c * z).powi(3);
          if v <= T::zero() {
            continue;
          }
          let u: T = T::sample_uniform(rng);
          let z2 = z * z;
          if u < T::one() - T::from(0.0331).unwrap() * z2 * z2 {
            break d * v;
          }
          if u.ln() < T::from(0.5).unwrap() * z2 + d * (T::one() - v + v.ln()) {
            break d * v;
          }
        };
        *x = self.scale * val;
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

impl<T: SimdFloatExt> Distribution<T> for SimdGamma<T> {
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

py_distribution!(PyGamma, SimdGamma,
  sig: (alpha, scale, dtype=None),
  params: (alpha: f64, scale: f64)
);
