use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloat;
use super::inverse_gauss::SimdInverseGauss;
use super::normal::SimdNormal;

pub struct SimdNormalInverseGauss<T: SimdFloat> {
  alpha: T,
  beta: T,
  delta: T,
  mu: T,
  ig: SimdInverseGauss<T>,
  normal: SimdNormal<T>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
}

impl<T: SimdFloat> SimdNormalInverseGauss<T> {
  pub fn new(alpha: T, beta: T, delta: T, mu: T) -> Self {
    assert!(
      alpha > T::zero() && alpha > beta.abs(),
      "NIG: alpha must be > |beta|"
    );
    assert!(delta > T::zero(), "NIG: delta must be positive");
    let gamma = (alpha * alpha - beta * beta).sqrt();
    let ig_mean = delta / gamma;
    let ig_shape = delta * delta;
    let ig = SimdInverseGauss::new(ig_mean, ig_shape);
    let normal = SimdNormal::new(T::zero(), T::one());
    Self {
      alpha,
      beta,
      delta,
      mu,
      ig,
      normal,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [T]) {
    let mut dbuf = vec![T::zero(); out.len()];
    let mut zbuf = vec![T::zero(); out.len()];
    self.ig.fill_slice(rng, &mut dbuf);
    self.normal.fill_slice(rng, &mut zbuf);

    let mu = T::splat(self.mu);
    let beta = T::splat(self.beta);

    let mut o_chunks = out.chunks_exact_mut(8);
    let mut d_chunks = dbuf.chunks_exact(8);
    let mut z_chunks = zbuf.chunks_exact(8);
    for ((co, cd), cz) in (&mut o_chunks).zip(&mut d_chunks).zip(&mut z_chunks) {
      let mut ad = [T::zero(); 8];
      ad.copy_from_slice(cd);
      let mut az = [T::zero(); 8];
      az.copy_from_slice(cz);
      let d = T::simd_from_array(ad);
      let z = T::simd_from_array(az);
      let x = mu + beta * d + T::simd_sqrt(d) * z;
      co.copy_from_slice(&T::simd_to_array(x));
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

impl<T: SimdFloat> Distribution<T> for SimdNormalInverseGauss<T> {
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
