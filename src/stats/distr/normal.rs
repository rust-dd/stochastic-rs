use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use wide::f32x8;

use super::fill_f32_zero_one;

/// A SIMD-based normal (Gaussian) random number generator using the wide crate.
/// It generates 16 normal samples at a time using the standard Box–Muller transform.
pub struct SimdNormal {
  mean: f32,
  std_dev: f32,
  /// Internal buffer holds 16 precomputed normal samples.
  buffer: UnsafeCell<[f32; 16]>,
  /// Current read index into `buffer`.
  index: UnsafeCell<usize>,
}

impl SimdNormal {
  /// Creates a new SimdNormal that will generate samples from N(mean, std_dev^2).
  pub fn new(mean: f32, std_dev: f32) -> Self {
    assert!(std_dev > 0.0);
    Self {
      mean,
      std_dev,
      buffer: UnsafeCell::new([0.0; 16]),
      index: UnsafeCell::new(16),
      // Start "full" so the first sample call triggers a refill.
    }
  }

  /// Refills the internal buffer with 16 samples using a vectorized Box–Muller transform.
  fn refill_buffer<R: Rng + ?Sized>(&self, rng: &mut R) {
    let buf = unsafe { &mut *self.buffer.get() };
    Self::fill_normal_f32x8(buf, rng);
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
  fn fill_normal_f32x8<R: Rng + ?Sized>(buf: &mut [f32; 16], rng: &mut R) {
    // Generate 8 random values for u1
    let mut arr_u1 = [0.0_f32; 8];
    fill_f32_zero_one(rng, &mut arr_u1);

    // Generate 8 random values for u2
    let mut arr_u2 = [0.0_f32; 8];
    fill_f32_zero_one(rng, &mut arr_u2);

    // Load them into f32x8 vectors
    let u1 = f32x8::from(arr_u1);
    let u2 = f32x8::from(arr_u2);

    // Box–Muller:
    // r = sqrt(-2 * ln(u1)), theta = 2 * PI * u2
    let neg_two = f32x8::splat(-2.0);
    let two_pi = f32x8::splat(2.0 * std::f32::consts::PI);

    let r = (neg_two * u1.ln()).sqrt();
    let theta = two_pi * u2;

    // Compute z0 = r * cos(theta)
    let z0 = r * theta.cos();
    // Compute z1 = r * sin(theta)
    let z1 = r * theta.sin();

    // Each lane i of z0, z1 is one normal sample.
    // So we have 16 total samples in two f32x8 vectors.

    let arr_z0 = z0.to_array(); // 8 floats
    let arr_z1 = z1.to_array(); // another 8 floats

    // Put z0 in the first 8 slots, z1 in the next 8 slots
    buf[..8].copy_from_slice(&arr_z0);
    buf[8..16].copy_from_slice(&arr_z1);
  }
}

impl Distribution<f32> for SimdNormal {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f32 {
    let index = unsafe { &mut *self.index.get() };
    if *index >= 16 {
      self.refill_buffer(rng);
    }
    let buf = unsafe { &mut *self.buffer.get() };
    let z = buf[*index];
    *index += 1;
    // Transform from standard normal to N(mean, std_dev^2)
    self.mean + self.std_dev * z
  }
}

#[cfg(test)]
mod tests {
  /// Tests the SimdNormal distribution by generating 100k samples and checking
  /// the empirical mean and variance, and plotting a histogram in the range [-4, 4].
  #[test]
  fn test_normal() {
    use super::*;
    use rand::thread_rng;

    let mut rng = thread_rng();
    let normal = SimdNormal::new(0.0, 1.0);

    // Generate 100k samples
    let n = 100_000;
    let samples: Vec<f32> = (0..n).map(|_| normal.sample(&mut rng)).collect();

    // Create a histogram in [-4, 4]
    let bins_count = 100;
    let min_x = -4.0;
    let max_x = 4.0;
    let bin_width = (max_x - min_x) / bins_count as f32;
    let mut bins = vec![0.0; bins_count];
    for &val in &samples {
      if val >= min_x && val < max_x {
        let idx = ((val - min_x) / bin_width) as usize;
        bins[idx] += 1.0;
      }
    }

    // Convert counts to a PDF-like estimate
    let total = samples.len() as f32;
    for b in bins.iter_mut() {
      *b /= total * bin_width;
    }

    // Bin centers
    let xs: Vec<f32> = (0..bins_count)
      .map(|i| min_x + (i as f32 + 0.5) * bin_width)
      .collect();

    // Plot the estimated PDF using plotly (optional)
    use plotly::{
      common::{Line, LineShape, Mode},
      Plot, Scatter,
    };
    let trace = Scatter::new(xs, bins)
      .mode(Mode::Lines)
      .line(Line::new().shape(LineShape::Linear))
      .name("Estimated PDF");
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.show();

    // Check empirical mean and variance
    let mean_sample = samples.iter().copied().sum::<f32>() / n as f32;
    let var_sample = samples
      .iter()
      .map(|&x| (x - mean_sample) * (x - mean_sample))
      .sum::<f32>()
      / (n as f32 - 1.0);

    println!(
      "Empirical mean: {:.4}, Empirical variance: {:.4}",
      mean_sample, var_sample
    );
    assert!(mean_sample.abs() < 0.05);
    assert!((var_sample - 1.0).abs() < 0.05);
  }
}
