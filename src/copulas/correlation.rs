//! # Correlation
//!
//! $$
//! \rho_{ij}=\frac{\operatorname{Cov}(X_i,X_j)}{\sigma_i\sigma_j}
//! $$
//!
use ndarray::Array2;

/// Kendall's tau matrix for a given data matrix
pub fn kendall_tau(data: &Array2<f64>) -> Array2<f64> {
  let cols = data.ncols();
  let mut tau_matrix = Array2::<f64>::zeros((cols, cols));

  for i in 0..cols {
    for j in i..cols {
      let col_i = data.column(i);
      let col_j = data.column(j);
      let mut concordant = 0;
      let mut discordant = 0;

      for k in 0..col_i.len() {
        for l in (k + 1)..col_i.len() {
          let x_diff = col_i[k] - col_i[l];
          let y_diff = col_j[k] - col_j[l];
          let sign = x_diff * y_diff;

          if sign > 0.0 {
            concordant += 1;
          } else if sign < 0.0 {
            discordant += 1;
          }
        }
      }

      let total_pairs = (col_i.len() * (col_i.len() - 1)) / 2;
      let tau = (concordant as f64 - discordant as f64) / total_pairs as f64;
      tau_matrix[[i, j]] = tau;
      tau_matrix[[j, i]] = tau;
    }
  }

  tau_matrix
}