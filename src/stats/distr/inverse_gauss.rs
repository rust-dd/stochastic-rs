use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use wide::f32x8;

use super::fill_f32_zero_one;
use super::normal::SimdNormal;

pub struct SimdInverseGauss {
  mu: f32,
  lambda: f32,
  normal: SimdNormal,
  buffer: UnsafeCell<[f32; 16]>,
  index: UnsafeCell<usize>,
}

impl SimdInverseGauss {
  pub fn new(mu: f32, lambda: f32) -> Self {
    assert!(mu > 0.0 && lambda > 0.0);
    Self {
      mu,
      lambda,
      normal: SimdNormal::new(0.0, 1.0),
      buffer: UnsafeCell::new([0.0; 16]),
      index: UnsafeCell::new(16),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [f32]) {
    // Get z normals and u uniforms in batches of 8
    let mut zbuf = vec![0.0f32; out.len()];
    self.normal.fill_slice(rng, &mut zbuf);
    let mut ubuf = vec![0.0f32; out.len()];
    // fill ubuf with uniforms via chunks
    let mut tmpu = [0.0f32; 8];
    let mut chunks = ubuf.chunks_exact_mut(8);
    for c in &mut chunks {
      fill_f32_zero_one(rng, &mut tmpu);
      c.copy_from_slice(&tmpu);
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      fill_f32_zero_one(rng, &mut tmpu);
      rem.copy_from_slice(&tmpu[..rem.len()]);
    }

    let mu = f32x8::splat(self.mu);
    let lam = f32x8::splat(self.lambda);

    let total = out.len();
    let mut out_chunks = out.chunks_exact_mut(8);
    let mut z_chunks = zbuf.chunks_exact(8);
    let mut u_chunks = ubuf.chunks_exact(8);
    for ((co, cz), cu) in (&mut out_chunks).zip(&mut z_chunks).zip(&mut u_chunks) {
      let mut az = [0.0f32; 8];
      az.copy_from_slice(cz);
      let mut au = [0.0f32; 8];
      au.copy_from_slice(cu);
      let z = f32x8::from(az);
      let u = f32x8::from(au);
      let w = z * z;
      let t1 = mu + (mu * mu * w) / (f32x8::splat(2.0) * lam);
      let rad = (f32x8::splat(4.0) * mu * lam * w + mu * mu * w * w).sqrt();
      let x = t1 - (mu / (f32x8::splat(2.0) * lam)) * rad;
      let check = mu / (mu + x);
      let alt = (mu * mu) / x;
      let ua = u.to_array();
      let xa = x.to_array();
      let ca = check.to_array();
      let aa = alt.to_array();
      for j in 0..8 {
        co[j] = if ua[j] < ca[j] { xa[j] } else { aa[j] };
      }
    }
    let ro = out_chunks.into_remainder();
    if !ro.is_empty() {
      // process tail scalar
      let base = total - ro.len();
      for i in 0..ro.len() {
        let z = zbuf[base + i];
        let u = ubuf[base + i];
        let w = z * z;
        let mu_s = self.mu;
        let lam_s = self.lambda;
        let t1 = mu_s + (mu_s * mu_s * w) / (2.0 * lam_s);
        let rad = (4.0 * mu_s * lam_s * w + mu_s * mu_s * w * w).sqrt();
        let x = t1 - (mu_s / (2.0 * lam_s)) * rad;
        let check = mu_s / (mu_s + x);
        ro[i] = if u < check { x } else { mu_s * mu_s / x };
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

impl Distribution<f32> for SimdInverseGauss {
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
