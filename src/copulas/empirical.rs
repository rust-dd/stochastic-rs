//! # Empirical
//!
//! $$
//! C_n(u,v)=\frac{1}{n}\sum_{i=1}^n \mathbf 1\{U_i\le u,\,V_i\le v\}
//! $$
//!
use ndarray::Array1;
use ndarray::Array2;

/// Empirical copula (2D) - rank-based transformation
#[derive(Clone, Debug)]
pub struct EmpiricalCopula2D {
  /// The rank-transformed data (N x 2), each row in [0,1]^2
  pub rank_data: Array2<f64>,
}

impl EmpiricalCopula2D {
  /// Create an EmpiricalCopula2D from two 1D arrays (`x` and `y`) of equal length.
  /// This performs a rank-based transform: for each sample i,
  ///   sx[i] = rank_of_x[i] / n
  ///   sy[i] = rank_of_y[i] / n
  /// and stores the resulting points in [0,1]^2.
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

  fn sample(&self, _n: usize) -> Array2<f64> {
    self.rank_data.clone()
  }
}