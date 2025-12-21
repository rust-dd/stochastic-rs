use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use wide::f64x8;

use super::fill_f64_zero_one;

/// A SIMD-based normal (Gaussian) random number generator using the wide crate.
/// It generates 16 normal samples at a time using the standard Box–Muller transform.
pub struct SimdNormal {
  mean: f64,
  std_dev: f64,
  /// Internal buffer holds 16 precomputed normal samples.
  buffer: UnsafeCell<[f64; 16]>,
  /// Current read index into `buffer`.
  index: UnsafeCell<usize>,
}

impl SimdNormal {
  /// Creates a new SimdNormal that will generate samples from N(mean, std_dev^2).
  pub fn new(mean: f64, std_dev: f64) -> Self {
    assert!(std_dev > 0.0);
    Self {
      mean,
      std_dev,
      buffer: UnsafeCell::new([0.0; 16]),
      index: UnsafeCell::new(16),
      // Start "full" so the first sample call triggers a refill.
    }
  }

  /// Efficiently fill `out` with samples using SIMD batches of 16.
  pub fn fill_slice<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [f64]) {
    let mut tmp = [0.0f64; 16];
    let mut chunks = out.chunks_exact_mut(16);
    for chunk in &mut chunks {
      Self::fill_normal_f64x8(&mut tmp, rng, self.mean, self.std_dev);
      chunk.copy_from_slice(&tmp);
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      Self::fill_normal_f64x8(&mut tmp, rng, self.mean, self.std_dev);
      rem.copy_from_slice(&tmp[..rem.len()]);
    }
  }

  /// Fill exactly 16 outputs into a mutable slice (must be length >= 16).
  #[inline]
  pub fn fill_16<R: Rng + ?Sized>(&self, rng: &mut R, out16: &mut [f64]) {
    debug_assert!(out16.len() >= 16);
    let mut tmp = [0.0f64; 16];
    Self::fill_normal_f64x8(&mut tmp, rng, self.mean, self.std_dev);
    out16[..16].copy_from_slice(&tmp);
  }

  /// Refills the internal buffer with 16 samples using a vectorized Box–Muller transform.
  fn refill_buffer<R: Rng + ?Sized>(&self, rng: &mut R) {
    let buf = unsafe { &mut *self.buffer.get() };
    Self::fill_normal_f64x8(buf, rng, self.mean, self.std_dev);
    unsafe {
      *self.index.get() = 0;
    }
  }

  /// Performs the standard Box–Muller transform on 8 pairs of uniform random values in [0, 1).
  /// Each pair (u1[i], u2[i]) produces two normal samples: z0[i], z1[i].
  ///
  /// - `u1` is used for the radius (r = sqrt(-2 ln(u1))).
  /// - `u2` is used for the angle (theta = 2 π u2).
  /// - We compute z0 = r cos(theta) and z1 = r sin(theta) for all 8 lanes.
  /// - That yields 16 total normal samples in `buf[0..16]`.
  /// - We also apply the affine transform N(0,1) -> N(mean, std_dev^2) in SIMD.
  fn fill_normal_f64x8<R: Rng + ?Sized>(buf: &mut [f64; 16], rng: &mut R, mean: f64, std_dev: f64) {
    // Generate 8 random values for u1
    let mut arr_u1 = [0.0_f64; 8];
    fill_f64_zero_one(rng, &mut arr_u1);

    // Generate 8 random values for u2
    let mut arr_u2 = [0.0_f64; 8];
    fill_f64_zero_one(rng, &mut arr_u2);

    // Load them into f64x8 vectors
    let mut u1 = f64x8::from(arr_u1);
    let u2 = f64x8::from(arr_u2);

    // Avoid ln(0)
    let eps = f64x8::splat(f64::MIN_POSITIVE);
    u1 = u1.max(eps);

    // Box–Muller:
    // r = sqrt(-2 * ln(u1)), theta = 2 * PI * u2
    let neg_two = f64x8::splat(-2.0);
    let two_pi = f64x8::splat(2.0 * std::f64::consts::PI);

    let r = (neg_two * u1.ln()).sqrt();
    let theta = two_pi * u2;

    // Compute z0 = r * cos(theta) and z1 = r * sin(theta)
    let z0 = r * theta.cos();
    let z1 = r * theta.sin();

    // Apply affine transform in SIMD: x = mean + std_dev * z
    let mm = f64x8::splat(mean);
    let ss = f64x8::splat(std_dev);
    let x0 = mm + ss * z0;
    let x1 = mm + ss * z1;

    let arr_x0 = x0.to_array();
    let arr_x1 = x1.to_array();

    // Put x0 in the first 8 slots, x1 in the next 8 slots
    buf[..8].copy_from_slice(&arr_x0);
    buf[8..16].copy_from_slice(&arr_x1);
  }
}

impl Distribution<f64> for SimdNormal {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
    let index = unsafe { &mut *self.index.get() };
    if *index >= 16 {
      self.refill_buffer(rng);
    }
    let buf = unsafe { &mut *self.buffer.get() };
    let z = buf[*index];
    *index += 1;
    z
  }
}
