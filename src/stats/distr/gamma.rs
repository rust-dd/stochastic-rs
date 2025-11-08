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
    assert!(alpha > 0.0 && scale > 0.0);
    Self {
      alpha,
      scale,
      buffer: UnsafeCell::new([0.0; 16]),
      index: UnsafeCell::new(16),
      normal: SimdNormal::new(0.0, 1.0),
    }
  }

  /// Bulk fill using Marsaglia–Tsang (for alpha >= 1) or Ahrens-Dieter (for alpha < 1)
  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [f32]) {
    if self.alpha < 1.0 {
      // For alpha < 1, use the transformation: if X ~ Gamma(alpha+1, scale), then X*U^(1/alpha) ~ Gamma(alpha, scale)
      let gamma_plus_one = SimdGamma::new(self.alpha + 1.0, self.scale);
      for x in out.iter_mut() {
        let g = gamma_plus_one.sample(rng);
        let u: f32 = rng.gen_range(0.0..1.0);
        *x = g * u.powf(1.0 / self.alpha);
      }
    } else {
      // Marsaglia-Tsang for alpha >= 1
      let d = self.alpha - 1.0 / 3.0;
      let c = 1.0 / (9.0 * d).sqrt();
      for x in out.iter_mut() {
        let val = loop {
          let z = self.normal.sample(rng);
          let v = (1.0 + c * z).powi(3);
          if v <= 0.0 {
            continue;
          }
          let u: f32 = rng.gen_range(0.0..1.0);
          let z2 = z * z;
          // Quick acceptance
          if u < 1.0 - 0.0331 * z2 * z2 {
            break d * v;
          }
          // Log acceptance
          if u.ln() < 0.5 * z2 + d * (1.0 - v + v.ln()) {
            break d * v;
          }
        };
        *x = self.scale * val;
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
