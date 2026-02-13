use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::inverse_gauss::SimdInverseGauss;
use super::normal::SimdNormal;
use super::SimdFloatExt;

pub struct SimdNormalInverseGauss<T: SimdFloatExt> {
  alpha: T,
  beta: T,
  delta: T,
  mu: T,
  ig: SimdInverseGauss<T>,
  normal: SimdNormal<T>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
}

impl<T: SimdFloatExt> SimdNormalInverseGauss<T> {
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
    let mu = T::splat(self.mu);
    let beta = T::splat(self.beta);
    let mut dbuf = [T::zero(); 8];
    let mut zbuf = [T::zero(); 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
      self.ig.fill_slice(rng, &mut dbuf);
      self.normal.fill_slice(rng, &mut zbuf);
      let d = T::simd_from_array(dbuf);
      let z = T::simd_from_array(zbuf);
      let x = mu + beta * d + T::simd_sqrt(d) * z;
      chunk.copy_from_slice(&T::simd_to_array(x));
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      self.ig.fill_slice(rng, &mut dbuf);
      self.normal.fill_slice(rng, &mut zbuf);
      for i in 0..rem.len() {
        let d = dbuf[i];
        let z = zbuf[i];
        rem[i] = self.mu + self.beta * d + d.sqrt() * z;
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

impl<T: SimdFloatExt> Distribution<T> for SimdNormalInverseGauss<T> {
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

py_distribution!(PyNormalInverseGauss, SimdNormalInverseGauss,
  sig: (alpha, beta, delta, mu, dtype=None),
  params: (alpha: f64, beta: f64, delta: f64, mu: f64)
);
