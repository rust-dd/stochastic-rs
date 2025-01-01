use ndarray::{Array1, Array2};
use plotly::{Plot, Scatter};
use rand::prelude::*;
use statrs::distribution::{ContinuousCDF, MultivariateNormal, Normal};

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

/// ========================================================================
/// Free functions for the *CDF* of Clayton and Gumbel (2D) from Wikipedia
/// ========================================================================

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

/// ========================================================================
/// 1) Empirical copula (2D) - rank-based transformation
/// ========================================================================
#[derive(Clone, Debug)]
pub struct EmpiricalCopula2D {
  /// The rank-transformed data (N x 2), each row in [0,1]^2
  pub rank_data: Array2<f64>,
}

impl EmpiricalCopula2D {
  /// Create an EmpiricalCopula2D from two 1D arrays (`xdata` and `ydata`) of equal length.
  /// This performs a rank-based transform: for each sample i,
  ///   sx[i] = rank_of_x[i] / n
  ///   sy[i] = rank_of_y[i] / n
  /// and stores the resulting points in [0,1]^2.
  pub fn new_from_two_series(xdata: &Array1<f64>, ydata: &Array1<f64>) -> Self {
    assert_eq!(
      xdata.len(),
      ydata.len(),
      "xdata and ydata must have the same length!"
    );
    let n = xdata.len();

    // Convert to Vec for easier sorting with indices
    let mut xv: Vec<(f64, usize)> = xdata.iter().enumerate().map(|(i, &val)| (val, i)).collect();
    let mut yv: Vec<(f64, usize)> = ydata.iter().enumerate().map(|(i, &val)| (val, i)).collect();

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
    // Demonstration: simply return the rank-transformed data
    // If you want bootstrap, you can draw with replacement here.
    self.rank_data.clone()
  }

  fn get_params(&self) -> Vec<f64> {
    // Empirical copula has no explicit parameters
    vec![]
  }
}

/// ========================================================================
/// 2) Gaussian copula (2D)
/// ========================================================================
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

/// ========================================================================
/// 3) Gumbel copula (2D) - CORRECT (Archimedean) sampling
/// ========================================================================
use rand_distr::{Distribution, Exp};
use statrs::distribution::Gamma;

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

/// ========================================================================
/// 4) Clayton copula (2D) - CORRECT (Archimedean) sampling
/// ========================================================================
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

/// ========================================================================
/// Tests / Examples: generate samples and plot them for Empirical, Gaussian,
/// Gumbel, and Clayton copulas. Use `cargo test -- --nocapture`.
/// ========================================================================
#[cfg(test)]
mod tests {
  use super::*;
  use ndarray::arr2;
  use rand_distr::Uniform;

  /// Number of samples for each copula
  const N: usize = 10000;

  /// ========================================================================
  /// 1) Empirical Copula Test
  /// ========================================================================
  #[test]
  fn test_empirical_copula() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(0.0, 1.0);

    let len_data = 500;
    let mut xdata = Array1::<f64>::zeros(len_data);
    let mut ydata = Array1::<f64>::zeros(len_data);
    for i in 0..len_data {
      let xv = uniform.sample(&mut rng);
      // Introduce some linear correlation
      let yv = 0.3 * uniform.sample(&mut rng) + 0.7 * xv;
      xdata[i] = xv;
      ydata[i] = yv.clamp(0.0, 1.0);
    }

    let empirical = EmpiricalCopula2D::new_from_two_series(&xdata, &ydata);
    let emp_samples = empirical.sample(N);
    plot_copula_samples(&emp_samples, "Empirical Copula (2D) - Rank-based data");
  }

  /// ========================================================================
  /// 2) Gaussian Copula Test
  /// ========================================================================
  #[test]
  fn test_gaussian_copula() {
    let gauss = GaussianCopula2D {
      mean: Array1::from(vec![0.0, 0.0]),
      cov: arr2(&[[1.0, 0.6], [0.6, 1.0]]),
    };
    let gauss_samples = gauss.sample(N);
    plot_copula_samples(&gauss_samples, "Gaussian Copula (2D)");
  }

  /// ========================================================================
  /// 3) Gumbel Copula Test
  /// ========================================================================
  #[test]
  fn test_gumbel_copula() {
    let gumbel = GumbelCopula2D { alpha: 4.0 };
    let gumbel_samples = gumbel.sample(N);
    plot_copula_samples(&gumbel_samples, "Gumbel Copula (2D) - Marshall–Olkin");

    // Example: Calculate the CDF of a specific point
    let c_gumb = cdf_gumbel(0.5, 0.8, 1.5);
    println!("Gumbel(θ=1.5) CDF(0.5, 0.8) = {}", c_gumb);
  }

  /// ========================================================================
  /// 4) Clayton Copula Test
  /// ========================================================================
  #[test]
  fn test_clayton_copula() {
    let clayton = ClaytonCopula2D { alpha: 1.5 };
    let clayton_samples = clayton.sample(N);
    plot_copula_samples(&clayton_samples, "Clayton Copula (2D) - Marshall–Olkin");

    // Example: Calculate the CDF of a specific point
    let c_clay = cdf_clayton(0.5, 0.8, 2.0);
    println!("Clayton(θ=2) CDF(0.5, 0.8) = {}", c_clay);
  }
}
