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

fn spearman_correlation(data: &Array2<f64>) -> Array2<f64> {
  let cols = data.ncols();
  let mut rho_matrix = Array2::<f64>::zeros((cols, cols));

  for i in 0..cols {
    for j in i..cols {
      let col_i = data.column(i);
      let col_j = data.column(j);

      let mean_i = col_i.sum() / col_i.len() as f64;
      let mean_j = col_j.sum() / col_j.len() as f64;

      let numerator: f64 = col_i
        .iter()
        .zip(col_j.iter())
        .map(|(&xi, &yi)| (xi - mean_i) * (yi - mean_j))
        .sum();

      let denominator_i = col_i
        .iter()
        .map(|&xi| (xi - mean_i).powi(2))
        .sum::<f64>()
        .sqrt();
      let denominator_j = col_j
        .iter()
        .map(|&yi| (yi - mean_j).powi(2))
        .sum::<f64>()
        .sqrt();

      let rho = numerator / (denominator_i * denominator_j);
      rho_matrix[[i, j]] = rho;
      rho_matrix[[j, i]] = rho; // Szimmetrikus mátrix
    }
  }

  rho_matrix
}
