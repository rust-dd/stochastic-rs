use ndarray::{Array1, Array2};
use plotly::{Plot, Scatter};
use rand::prelude::*;
use statrs::distribution::{ContinuousCDF, Exp, Gamma, MultivariateNormal, Normal};

/// A simple trait for 2D copulas: requires only a `sample` method and `get_params`.
pub trait NCopula2D {
  /// Generate `n` samples in [0,1]^2, returning them as an (n x 2) matrix.
  fn sample(&self, n: usize) -> Array2<f64>;

  /// Return the copula parameters as a `Vec<f64>`.
  fn get_params(&self) -> Vec<f64>;
}

/// A small helper function for plotting 2D data using Plotly.
pub fn plot_copula_samples(data: &Array2<f64>, title: &str) {
  if data.ncols() != 2 {
    eprintln!(
      "Only 2D data can be plotted, but got {} columns!",
      data.ncols()
    );
    return;
  }
  let x = data.column(0).to_vec();
  let y = data.column(1).to_vec();

  let trace = Scatter::new(x, y)
    .mode(plotly::common::Mode::Markers)
    .marker(plotly::common::Marker::new().size(3))
    .name(title);

  let mut plot = Plot::new();
  plot.add_trace(trace);
  plot.show();
}

/// Clayton copula CDF, θ in (-1,∞)\{0}:
/// C(u,v) = max(u^-θ + v^-θ - 1, 0)^(-1/θ)
pub fn cdf_clayton(u: f64, v: f64, theta: f64) -> f64 {
  let val = u.powf(-theta) + v.powf(-theta) - 1.0;
  val.max(0.0).powf(-1.0 / theta)
}

/// Gumbel copula CDF, θ in [1,∞):
/// C(u,v) = exp(-( (-ln(u))^θ + (-ln(v))^θ )^(1/θ))
pub fn cdf_gumbel(u: f64, v: f64, theta: f64) -> f64 {
  let s = (-u.ln()).powf(theta) + (-v.ln()).powf(theta);
  ((-1.0) * s.powf(1.0 / theta)).exp()
}

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
}

impl NCopula2D for EmpiricalCopula2D {
  fn sample(&self, _n: usize) -> Array2<f64> {
    self.rank_data.clone()
  }

  fn get_params(&self) -> Vec<f64> {
    vec![]
  }
}

/// Gaussian copula (2D)
#[derive(Clone, Debug)]
pub struct GaussianCopula2D {
  /// 2D mean vector, e.g. [0.0, 0.0]
  pub mean: Array1<f64>,
  /// 2x2 covariance matrix
  pub cov: Array2<f64>,
}

impl NCopula2D for GaussianCopula2D {
  fn sample(&self, n: usize) -> Array2<f64> {
    // Flatten the 2x2 covariance
    let mut cov_flat = Vec::with_capacity(4);
    cov_flat.push(self.cov[[0, 0]]);
    cov_flat.push(self.cov[[0, 1]]);
    cov_flat.push(self.cov[[1, 0]]);
    cov_flat.push(self.cov[[1, 1]]);

    // Create a 2D MVN
    let mvn = MultivariateNormal::new(self.mean.to_vec(), cov_flat)
      .expect("Invalid MVN parameters (Gaussian copula).");

    let mut rng = thread_rng();
    let mut z = Array2::<f64>::zeros((n, 2));

    // Sample from MVN
    for i in 0..n {
      let sample_vec = mvn.sample(&mut rng);
      z[[i, 0]] = sample_vec[0];
      z[[i, 1]] = sample_vec[1];
    }

    // Apply standard normal CDF to get each coordinate in [0,1]
    let std_normal = Normal::new(0.0, 1.0).unwrap();
    for i in 0..n {
      z[[i, 0]] = std_normal.cdf(z[[i, 0]]);
      z[[i, 1]] = std_normal.cdf(z[[i, 1]]);
    }
    z
  }

  fn get_params(&self) -> Vec<f64> {
    vec![
      self.mean[0],
      self.mean[1],
      self.cov[[0, 0]],
      self.cov[[0, 1]],
      self.cov[[1, 0]],
      self.cov[[1, 1]],
    ]
  }
}

/// Gumbel copula (2D) - CORRECT (Archimedean) sampling
#[derive(Clone, Debug)]
pub struct GumbelCopula2D {
  /// alpha >= 1 (Gumbel parameter)
  pub alpha: f64,
}

impl NCopula2D for GumbelCopula2D {
  fn sample(&self, n: usize) -> Array2<f64> {
    let alpha = self.alpha;
    assert!(alpha >= 1.0, "The Gumbel parameter (alpha) must be >= 1!");

    let mut rng = thread_rng();
    let exp_dist = Exp::new(1.0).unwrap(); // Exp(1)
    let mut data = Array2::<f64>::zeros((n, 2));

    for i in 0..n {
      // 1) M = X^alpha, where X ~ Exp(1)
      let x = exp_dist.sample(&mut rng) as f64;
      let m = x.powf(alpha);

      // 2) E1, E2 ~ Exp(1)
      let e1 = exp_dist.sample(&mut rng);
      let e2 = exp_dist.sample(&mut rng);

      // 3) U1 = exp(- (E1*M)^(1/alpha)), U2 = exp(- (E2*M)^(1/alpha))
      let u1 = (-(e1 * m).powf(1.0 / alpha)).exp();
      let u2 = (-(e2 * m).powf(1.0 / alpha)).exp();

      data[[i, 0]] = u1;
      data[[i, 1]] = u2;
    }
    data
  }

  fn get_params(&self) -> Vec<f64> {
    vec![self.alpha]
  }
}

/// Clayton copula (2D) - CORRECT (Archimedean) sampling
#[derive(Clone, Debug)]
pub struct ClaytonCopula2D {
  /// alpha > 0 (Clayton parameter)
  pub alpha: f64,
}

impl NCopula2D for ClaytonCopula2D {
  fn sample(&self, n: usize) -> Array2<f64> {
    let alpha = self.alpha;
    assert!(alpha > 0.0, "The Clayton parameter (alpha) must be > 0!");

    let mut rng = thread_rng();
    // Gamma(shape = 1/alpha, rate = 1)
    let gamma_dist = Gamma::new(1.0 / alpha, 1.0).unwrap();
    let exp_dist = Exp::new(1.0).unwrap(); // Exp(1)

    let mut data = Array2::<f64>::zeros((n, 2));

    for i in 0..n {
      // 1) W ~ Gamma(shape=1/alpha, rate=1)
      let w = gamma_dist.sample(&mut rng);

      // 2) E1, E2 ~ Exp(1)
      let e1 = exp_dist.sample(&mut rng);
      let e2 = exp_dist.sample(&mut rng);

      // 3) U1 = (1 + W*E1)^(-1/alpha), U2 = (1 + W*E2)^(-1/alpha)
      let u1 = (1.0 + w * e1).powf(-1.0 / alpha);
      let u2 = (1.0 + w * e2).powf(-1.0 / alpha);

      data[[i, 0]] = u1;
      data[[i, 1]] = u2;
    }
    data
  }

  fn get_params(&self) -> Vec<f64> {
    vec![self.alpha]
  }
}

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

#[cfg(test)]
mod tests {
  use super::*;
  use ndarray::arr2;
  use rand_distr::Uniform;

  const N: usize = 10000;

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

  #[test]
  fn test_gaussian_copula() {
    let gauss = GaussianCopula2D {
      mean: Array1::from(vec![0.0, 0.0]),
      cov: arr2(&[[1.0, 0.6], [0.6, 1.0]]),
    };
    let gauss_samples = gauss.sample(N);
    plot_copula_samples(&gauss_samples, "Gaussian Copula (2D)");
  }

  #[test]
  fn test_gumbel_copula() {
    let gumbel = GumbelCopula2D { alpha: 4.0 };
    let gumbel_samples = gumbel.sample(N);
    plot_copula_samples(&gumbel_samples, "Gumbel Copula (2D) - Marshall–Olkin");

    // Example: Calculate the CDF of a specific point
    let c_gumb = cdf_gumbel(0.5, 0.8, 1.5);
    println!("Gumbel(θ=1.5) CDF(0.5, 0.8) = {}", c_gumb);
  }

  #[test]
  fn test_clayton_copula() {
    let clayton = ClaytonCopula2D { alpha: 1.5 };
    let clayton_samples = clayton.sample(N);
    plot_copula_samples(&clayton_samples, "Clayton Copula (2D) - Marshall–Olkin");

    // Example: Calculate the CDF of a specific point
    let c_clay = cdf_clayton(0.5, 0.8, 2.0);
    println!("Clayton(θ=2) CDF(0.5, 0.8) = {}", c_clay);
  }

  #[test]
  fn test_kendall_tau() {
    let data = arr2(&[
      [1.0, 2.0, 3.0],
      [2.0, 3.0, 1.0],
      [3.0, 1.0, 2.0],
      [4.0, 4.0, 4.0],
    ]);
    let x = data.column(0).to_owned();
    let y = data.column(1).to_owned();
    let copula = EmpiricalCopula2D::new_from_two_series(&x, &y);
    let tau_matrix = kendall_tau(&copula.rank_data);
    println!("Kendall's tau matrix:\n{:?}", tau_matrix);
  }

  #[test]
  fn test_spearman_correlation() {
    let data = arr2(&[
      [1.0, 2.0, 3.0],
      [2.0, 3.0, 1.0],
      [3.0, 1.0, 2.0],
      [4.0, 4.0, 4.0],
    ]);
    let x = data.column(0).to_owned();
    let y = data.column(1).to_owned();
    let copula = EmpiricalCopula2D::new_from_two_series(&x, &y);
    let rho_matrix = spearman_correlation(&copula.rank_data);
    println!("Spearman's rho matrix:\n{:?}", rho_matrix);
  }
}
