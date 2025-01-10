use ndarray::{Array1, Array2};

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

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use rand::thread_rng;
  use rand_distr::{Distribution, Uniform};

  use crate::{stats::copula::plot_copula_samples, stochastic::N};

  use super::EmpiricalCopula2D;

  #[test]
  fn test_empirical_copula() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(0.0, 1.0);

    let len_data = 500;
    let mut x = Array1::<f64>::zeros(len_data);
    let mut y = Array1::<f64>::zeros(len_data);
    for i in 0..len_data {
      let xv = uniform.sample(&mut rng);
      // Introduce some linear correlation
      let yv = 0.3 * uniform.sample(&mut rng) + 0.7 * xv;
      x[i] = xv;
      y[i] = yv.clamp(0.0, 1.0);
    }

    let empirical = EmpiricalCopula2D::new_from_two_series(&x, &y);
    let emp_samples = empirical.sample(N);
    plot_copula_samples(&emp_samples, "Empirical Copula (2D) - Rank-based data");
  }
}
