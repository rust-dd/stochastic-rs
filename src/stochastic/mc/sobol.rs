//! # Sobol Sequence
//!
//! $$
//! x_n = \bigoplus_{k:\,b_k(n)=1} v_k,\quad
//! n = \sum_k b_k(n)\,2^k
//! $$
//!
//! Gray-code Sobol sequence using Joe-Kuo direction numbers.
//!
//! Reference: Joe & Kuo (2010), "Constructing Sobol Sequences with Better
//! Two-Dimensional Projections", DOI: 10.1137/070709359

use ndarray::Array2;

use crate::traits::FloatExt;

const BITS: usize = 32;

/// Joe-Kuo direction number data for dimensions 2..=21.
///
/// Format: `(s, a, m_i)` where `s` is the degree of the primitive polynomial,
/// `a` encodes the middle coefficients, and `m_i` are the initial direction
/// numbers.
const JOE_KUO: &[(u32, u32, &[u32])] = &[
  (1, 0, &[1]),                    // dim 2
  (2, 1, &[1, 1]),                 // dim 3
  (3, 1, &[1, 1, 1]),              // dim 4
  (3, 2, &[1, 3, 1]),              // dim 5
  (4, 1, &[1, 1, 3, 3]),           // dim 6
  (4, 4, &[1, 3, 5, 13]),          // dim 7
  (5, 2, &[1, 1, 5, 5, 17]),       // dim 8
  (5, 4, &[1, 1, 5, 5, 5]),        // dim 9
  (5, 7, &[1, 1, 7, 11, 19]),      // dim 10
  (5, 11, &[1, 1, 5, 1, 1]),       // dim 11
  (5, 13, &[1, 1, 1, 3, 11]),      // dim 12
  (5, 14, &[1, 3, 5, 5, 31]),      // dim 13
  (6, 1, &[1, 3, 3, 9, 7, 49]),    // dim 14
  (6, 13, &[1, 1, 1, 15, 21, 21]), // dim 15
  (6, 16, &[1, 3, 1, 13, 27, 49]), // dim 16
  (6, 19, &[1, 1, 1, 15, 7, 5]),   // dim 17
  (6, 22, &[1, 3, 1, 3, 29, 43]),  // dim 18
  (6, 25, &[1, 3, 5, 7, 11, 27]),  // dim 19
  (6, 34, &[1, 3, 1, 7, 3, 29]),   // dim 20
  (6, 37, &[1, 3, 3, 7, 7, 21]),   // dim 21
];

/// Compute 32-bit direction numbers from the primitive polynomial data.
fn compute_direction_numbers(s: u32, a: u32, m_init: &[u32]) -> [u32; BITS] {
  let s = s as usize;
  let mut v = [0u32; BITS];

  for j in 0..s {
    v[j] = m_init[j] << (BITS - 1 - j);
  }

  for j in s..BITS {
    let mut val = v[j - s] ^ (v[j - s] >> s as u32);
    for k in 1..s {
      let c_k = (a >> (s as u32 - 1 - k as u32)) & 1;
      if c_k == 1 {
        val ^= v[j - k];
      }
    }
    v[j] = val;
  }

  v
}

/// Sobol low-discrepancy sequence generator.
#[derive(Debug, Clone)]
pub struct SobolSeq {
  n_dims: usize,
  direction: Vec<[u32; BITS]>,
}

impl SobolSeq {
  /// Create a Sobol sequence generator for `n_dims` dimensions (max 21).
  pub fn new(n_dims: usize) -> Self {
    assert!(
      n_dims > 0 && n_dims <= JOE_KUO.len() + 1,
      "Sobol supports 1..={} dimensions, got {n_dims}",
      JOE_KUO.len() + 1
    );

    let mut direction = Vec::with_capacity(n_dims);

    // Dimension 1: van der Corput base 2
    let mut v0 = [0u32; BITS];
    for (j, v) in v0.iter_mut().enumerate() {
      *v = 1u32 << (BITS - 1 - j);
    }
    direction.push(v0);

    // Dimensions 2..=n_dims
    for d in 1..n_dims {
      let &(s, a, m) = &JOE_KUO[d - 1];
      direction.push(compute_direction_numbers(s, a, m));
    }

    Self { n_dims, direction }
  }

  /// Generate `n_points` Sobol points in `[0, 1)^d` (Gray-code order).
  ///
  /// Returns an `(n_points, n_dims)` array.
  pub fn sample<T: FloatExt>(&self, n_points: usize) -> Array2<T> {
    let scale = T::from_f64_fast(1.0 / (1u64 << BITS) as f64);
    let mut out = Array2::<T>::zeros((n_points, self.n_dims));
    let mut x = vec![0u32; self.n_dims];

    for i in 0..n_points {
      let c = ((i + 1) as u32).trailing_zeros() as usize;
      for j in 0..self.n_dims {
        x[j] ^= self.direction[j][c.min(BITS - 1)];
        out[[i, j]] = T::from_f64_fast(x[j] as f64) * scale;
      }
    }

    out
  }

  pub fn n_dims(&self) -> usize {
    self.n_dims
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn sobol_dim1_first_points() {
    let seq = SobolSeq::new(1);
    let pts: Array2<f64> = seq.sample(3);
    // Gray-code order: 0.5, 0.75, 0.25
    assert!((pts[[0, 0]] - 0.5).abs() < 1e-10);
    assert!((pts[[1, 0]] - 0.75).abs() < 1e-10);
    assert!((pts[[2, 0]] - 0.25).abs() < 1e-10);
  }

  #[test]
  fn sobol_points_in_unit_cube() {
    let seq = SobolSeq::new(10);
    let pts: Array2<f64> = seq.sample(1000);
    for i in 0..1000 {
      for j in 0..10 {
        assert!(pts[[i, j]] >= 0.0 && pts[[i, j]] < 1.0);
      }
    }
  }

  #[test]
  fn sobol_mean_converges_to_half() {
    let seq = SobolSeq::new(3);
    let n = 1023; // 2^10 − 1 for best uniformity
    let pts: Array2<f64> = seq.sample(n);
    for j in 0..3 {
      let mean: f64 = (0..n).map(|i| pts[[i, j]]).sum::<f64>() / n as f64;
      assert!(
        (mean - 0.5).abs() < 0.02,
        "dim {j} mean = {mean:.4}, expected ≈ 0.5"
      );
    }
  }
}
