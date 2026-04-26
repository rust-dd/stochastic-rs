//! # Halton Sequence
//!
//! $$
//! x_n^{(j)} = \sum_{k=0}^{\infty} d_k(n)\,p_j^{-(k+1)},\quad
//! n = \sum_{k=0}^{\infty} d_k(n)\,p_j^k
//! $$
//!
//! Van der Corput sequences in successive prime bases.
//!
//! Reference: Halton (1964), DOI: 10.1007/BF01386213

use ndarray::Array2;

use crate::traits::FloatExt;

const PRIMES: [usize; 40] = [
  2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
  101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
];

/// Van der Corput radical-inverse function in the given base.
fn van_der_corput(mut index: usize, base: usize) -> f64 {
  let mut result = 0.0;
  let mut f = 1.0 / base as f64;
  while index > 0 {
    result += (index % base) as f64 * f;
    index /= base;
    f /= base as f64;
  }
  result
}

/// Halton low-discrepancy sequence generator.
#[derive(Debug, Clone)]
pub struct HaltonSeq {
  n_dims: usize,
}

impl HaltonSeq {
  /// Create a Halton sequence generator for `n_dims` dimensions (max 40).
  pub fn new(n_dims: usize) -> Self {
    assert!(
      n_dims > 0 && n_dims <= PRIMES.len(),
      "Halton supports 1..={} dimensions, got {n_dims}",
      PRIMES.len()
    );
    Self { n_dims }
  }

  /// Generate `n_points` Halton points in `[0, 1)^d`.
  ///
  /// Returns an `(n_points, n_dims)` array. Points are 1-indexed
  /// (skipping the origin).
  pub fn sample<T: FloatExt>(&self, n_points: usize) -> Array2<T> {
    let mut out = Array2::<T>::zeros((n_points, self.n_dims));
    for i in 0..n_points {
      for j in 0..self.n_dims {
        out[[i, j]] = T::from_f64_fast(van_der_corput(i + 1, PRIMES[j]));
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
  fn halton_dim1_is_van_der_corput_base2() {
    let seq = HaltonSeq::new(1);
    let pts: Array2<f64> = seq.sample(4);
    let expected = [0.5, 0.25, 0.75, 0.125];
    for (i, &e) in expected.iter().enumerate() {
      assert!(
        (pts[[i, 0]] - e).abs() < 1e-12,
        "point {i}: got {}, expected {e}",
        pts[[i, 0]]
      );
    }
  }

  #[test]
  fn halton_points_in_unit_cube() {
    let seq = HaltonSeq::new(5);
    let pts: Array2<f64> = seq.sample(1000);
    for i in 0..1000 {
      for j in 0..5 {
        assert!(pts[[i, j]] > 0.0 && pts[[i, j]] < 1.0);
      }
    }
  }
}
