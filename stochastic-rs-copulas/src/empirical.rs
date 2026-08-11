//! # Empirical
//!
//! $$
//! C_n(u,v)=\frac{1}{n}\sum_{i=1}^n \mathbf 1\{U_i\le u,\,V_i\le v\}
//! $$
//!
use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::uniform::SimdUniform;

/// Empirical copula (2D) - rank-based transformation
#[derive(Clone, Debug)]
pub struct EmpiricalCopula2D {
  /// The rank-transformed data `(N x 2)`, each row in `[0, 1]^2`.
  pub rank_data: Array2<f64>,
}

impl EmpiricalCopula2D {
  /// Create an EmpiricalCopula2D from two 1D arrays (`x` and `y`) of equal length.
  /// This performs a rank-based transform: for each sample `i`,
  /// `sx_i = rank_of_x_i / n` and `sy_i = rank_of_y_i / n`,
  /// then stores the resulting points in `[0, 1]^2`.
  pub fn new_from_two_series(x: &Array1<f64>, y: &Array1<f64>) -> Self {
    assert_eq!(x.len(), y.len(), "x and y must have the same length!");
    let n = x.len();

    // Convert to Vec for easier sorting with indices
    let mut xv: Vec<(f64, usize)> = x.iter().enumerate().map(|(i, &val)| (val, i)).collect();
    let mut yv: Vec<(f64, usize)> = y.iter().enumerate().map(|(i, &val)| (val, i)).collect();

    // Sort by the actual float value
    xv.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    yv.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // After sorting, xv[k] = (value, original_index).
    // The rank of that original index is k.
    let mut rank_x = vec![0.0; n];
    let mut rank_y = vec![0.0; n];
    for (rank, &(_val, orig_i)) in xv.iter().enumerate() {
      rank_x[orig_i] = rank as f64; // rank in [0..n-1]
    }
    for (rank, &(_val, orig_i)) in yv.iter().enumerate() {
      rank_y[orig_i] = rank as f64;
    }

    // Normalize ranks to [0,1].
    for i in 0..n {
      rank_x[i] /= n as f64;
      rank_y[i] /= n as f64;
    }

    // Build final (n x 2) array
    let mut rank_data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
      rank_data[[i, 0]] = rank_x[i];
      rank_data[[i, 1]] = rank_y[i];
    }
    EmpiricalCopula2D { rank_data }
  }

  /// Bootstrap resample of `n` `(u, v)` rows drawn **with replacement**
  /// from the rank-transformed empirical support, using the crate's
  /// shared unseeded entropy stream (same default-randomness path as
  /// e.g. [`crate::bivariate::independence::Independence::sample`]).
  ///
  /// This does not draw from the continuous copula; it resamples the `n`
  /// observed `(u, v)` pairs, which is the standard nonparametric
  /// bootstrap for an empirical copula (Deheuvels, 1979).
  pub fn sample(&self, n: usize) -> Array2<f64> {
    self.sample_with_uniform(SimdUniform::<f64>::new(0.0, 1.0, &Unseeded), n)
  }

  /// Deterministic counterpart of [`EmpiricalCopula2D::sample`]: the same
  /// `seed` reproduces the same `n` bootstrap draws.
  pub fn sample_with_seed(&self, n: usize, seed: u64) -> Array2<f64> {
    self.sample_with_uniform(
      SimdUniform::<f64>::new(0.0, 1.0, &Deterministic::new(seed)),
      n,
    )
  }

  fn sample_with_uniform(&self, ud: SimdUniform<f64>, n: usize) -> Array2<f64> {
    let n_rows = self.rank_data.nrows();
    let mut out = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
      let idx = ((ud.sample_fast() * n_rows as f64) as usize).min(n_rows - 1);
      out[[i, 0]] = self.rank_data[[idx, 0]];
      out[[i, 1]] = self.rank_data[[idx, 1]];
    }
    out
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn fixture() -> EmpiricalCopula2D {
    let x = Array1::from_vec((0..50).map(|i| i as f64).collect());
    let y = Array1::from_vec((0..50).map(|i| (i as f64 * 1.7) % 37.0).collect());
    EmpiricalCopula2D::new_from_two_series(&x, &y)
  }

  /// Bootstrap resample: length equals `n` (not the original support size),
  /// two calls with the same seed are identical, and every value stays in
  /// the rank-transform's `[0, (n_support-1)/n_support]` image.
  #[test]
  fn empirical_sample_honors_n_and_seed() {
    let ec = fixture();
    let n = 17usize;
    let a = ec.sample_with_seed(n, 42);
    let b = ec.sample_with_seed(n, 42);
    assert_eq!(a.nrows(), n);
    assert_eq!(b.nrows(), n);
    assert_eq!(a, b, "same seed must reproduce identical draws");
    for row in a.rows() {
      for &val in row.iter() {
        assert!((0.0..1.0).contains(&val), "value {val} out of [0, 1)");
      }
    }

    let c = ec.sample_with_seed(n, 7);
    assert_ne!(a, c, "different seeds should (almost surely) differ");

    let unseeded = ec.sample(n);
    assert_eq!(unseeded.nrows(), n);
  }
}
