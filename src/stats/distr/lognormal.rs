use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use wide::f32x8;

use super::normal::SimdNormal;

pub struct SimdLogNormal {
  mu: f32,
  sigma: f32,
  buffer: UnsafeCell<[f32; 16]>,
  index: UnsafeCell<usize>,
  normal: SimdNormal,
}

impl SimdLogNormal {
  pub fn new(mu: f32, sigma: f32) -> Self {
    assert!(sigma > 0.0);
    Self {
      mu,
      sigma,
      buffer: UnsafeCell::new([0.0; 16]),
      index: UnsafeCell::new(16),
      normal: SimdNormal::new(0.0, 1.0),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [f32]) {
    // Generate normals in batches of 16
    let mut tmp = vec![0.0f32; out.len()];
    self.normal.fill_slice(rng, &mut tmp);

    // Apply affine and exp in SIMD over 8-lane chunks
    let mm = f32x8::splat(self.mu);
    let ss = f32x8::splat(self.sigma);

    let mut chunks_out = out.chunks_exact_mut(8);
    let mut chunks_in = tmp.chunks_exact(8);
    for (chunk_o, chunk_i) in (&mut chunks_out).zip(&mut chunks_in) {
      let mut a = [0.0f32; 8];
      a.copy_from_slice(chunk_i);
      let z = f32x8::from(a);
      let x = (mm + ss * z).exp();
      chunk_o.copy_from_slice(&x.to_array());
    }
    let rem_o = chunks_out.into_remainder();
    let rem_i = chunks_in.remainder();
    if !rem_o.is_empty() {
      let mut a = [0.0f32; 8];
      a[..rem_i.len()].copy_from_slice(rem_i);
      let z = f32x8::from(a);
      let x = (mm + ss * z).exp().to_array();
      rem_o.copy_from_slice(&x[..rem_o.len()]);
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

impl Distribution<f32> for SimdLogNormal {
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
