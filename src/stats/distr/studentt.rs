use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use wide::f32x8;

use super::{chi_square::SimdChiSquared, normal::SimdNormal};

pub struct SimdStudentT {
  nu: f32,
  normal: SimdNormal,
  chisq: SimdChiSquared,
  buffer: UnsafeCell<[f32; 16]>,
  index: UnsafeCell<usize>,
}

impl SimdStudentT {
  pub fn new(nu: f32) -> Self {
    Self {
      nu,
      normal: SimdNormal::new(0.0, 1.0),
      chisq: SimdChiSquared::new(nu),
      buffer: UnsafeCell::new([0.0; 16]),
      index: UnsafeCell::new(16),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [f32]) {
    let mut zbuf = vec![0.0f32; out.len()];
    let mut vbuf = vec![0.0f32; out.len()];
    self.normal.fill_slice(rng, &mut zbuf);
    self.chisq.fill_slice(rng, &mut vbuf);

    let inv_nu = f32x8::splat(1.0 / self.nu);

    let mut o_chunks = out.chunks_exact_mut(8);
    let mut z_chunks = zbuf.chunks_exact(8);
    let mut v_chunks = vbuf.chunks_exact(8);
    for ((co, cz), cv) in (&mut o_chunks).zip(&mut z_chunks).zip(&mut v_chunks) {
      let mut az = [0.0f32; 8];
      az.copy_from_slice(cz);
      let mut av = [0.0f32; 8];
      av.copy_from_slice(cv);
      let z = f32x8::from(az);
      let v = f32x8::from(av);
      let denom = (v * inv_nu).sqrt();
      let x = z / denom;
      co.copy_from_slice(&x.to_array());
    }
    let rem_o = o_chunks.into_remainder();
    let rem_z = z_chunks.remainder();
    let rem_v = v_chunks.remainder();
    if !rem_o.is_empty() {
      for i in 0..rem_o.len() {
        rem_o[i] = rem_z[i] / (rem_v[i] / self.nu).sqrt();
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

impl Distribution<f32> for SimdStudentT {
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
