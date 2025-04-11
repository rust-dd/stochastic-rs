use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::normal::SimdNormal;

pub struct SimdGamma {
  alpha: f32,
  scale: f32,
  buffer: UnsafeCell<[f32; 16]>,
  index: UnsafeCell<usize>,
  normal: SimdNormal,
}

impl SimdGamma {
  pub fn new(alpha: f32, scale: f32) -> Self {
    assert!(alpha >= 1.0 && scale > 0.0);
    Self {
      alpha,
      scale,
      buffer: UnsafeCell::new([0.0; 16]),
      index: UnsafeCell::new(16),
      normal: SimdNormal::new(0.0, 1.0),
    }
  }

  fn refill_buffer<R: Rng + ?Sized>(&self, rng: &mut R) {
    let buf = unsafe { &mut *self.buffer.get() };
    let d = self.alpha - 1.0 / 3.0;
    let c = 1.0 / (3.0 * d).sqrt();

    for i in 0..16 {
      // Marsaglia–Tsang
      let val = loop {
        let z = self.normal.sample(rng);
        let v = (1.0 + c * z).powi(3);
        if v <= 0.0 {
          continue;
        }
        let u: f32 = rng.gen_range(0.0..1.0);
        if u < 1.0 - 0.0331 * z.powi(4) {
          break self.scale * d * v;
        }
        if u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
          break self.scale * d * v;
        }
      };
      buf[i] = val;
    }
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl Distribution<f32> for SimdGamma {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f32 {
    let idx = unsafe { &mut *self.index.get() };
    if *idx >= 16 {
      self.refill_buffer(rng);
    }
    let val = unsafe { (*self.buffer.get())[*idx] };
    *idx += 1;
    val
  }
}
