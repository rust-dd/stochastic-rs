use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use wide::f32x8;

use super::{inverse_gauss::SimdInverseGauss, normal::SimdNormal};

pub struct SimdNormalInverseGauss {
  alpha: f32,
  beta: f32,
  delta: f32,
  mu: f32,
  ig: SimdInverseGauss,
  normal: SimdNormal,
  buffer: UnsafeCell<[f32; 16]>,
  index: UnsafeCell<usize>,
}

impl SimdNormalInverseGauss {
  pub fn new(alpha: f32, beta: f32, delta: f32, mu: f32) -> Self {
    // Typically alpha> |beta|, delta>0, etc.
    let ig = SimdInverseGauss::new(delta * (alpha * alpha - beta * beta).sqrt(), delta);
    let normal = SimdNormal::new(0.0, 1.0);
    Self {
      alpha,
      beta,
      delta,
      mu,
      ig,
      normal,
      buffer: UnsafeCell::new([0.0; 16]),
      index: UnsafeCell::new(16),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [f32]) {
    let mut dbuf = vec![0.0f32; out.len()];
    let mut zbuf = vec![0.0f32; out.len()];
    self.ig.fill_slice(rng, &mut dbuf);
    self.normal.fill_slice(rng, &mut zbuf);

    let mu = f32x8::splat(self.mu);
    let beta = f32x8::splat(self.beta);

    let mut o_chunks = out.chunks_exact_mut(8);
    let mut d_chunks = dbuf.chunks_exact(8);
    let mut z_chunks = zbuf.chunks_exact(8);
    for ((co, cd), cz) in (&mut o_chunks).zip(&mut d_chunks).zip(&mut z_chunks) {
      let mut ad = [0.0f32; 8];
      ad.copy_from_slice(cd);
      let mut az = [0.0f32; 8];
      az.copy_from_slice(cz);
      let d = f32x8::from(ad);
      let z = f32x8::from(az);
      let x = mu + beta * d + d.sqrt() * z;
      co.copy_from_slice(&x.to_array());
    }
    let rem_o = o_chunks.into_remainder();
    let rem_d = d_chunks.remainder();
    let rem_z = z_chunks.remainder();
    if !rem_o.is_empty() {
      for i in 0..rem_o.len() {
        let d = rem_d[i];
        let z = rem_z[i];
        rem_o[i] = self.mu + self.beta * d + d.sqrt() * z;
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

impl Distribution<f32> for SimdNormalInverseGauss {
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
