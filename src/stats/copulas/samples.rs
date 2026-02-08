use ndarray::Array1;
use ndarray::Array2;
use plotly::Plot;
use plotly::Scatter;
use rand::prelude::*;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Exp;
use statrs::distribution::Gamma;
use statrs::distribution::MultivariateNormal;
use statrs::distribution::Normal;

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

    let mut rng = rand::rng();
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

    let mut rng = rand::rng();
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

    let mut rng = rand::rng();
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

#[cfg(test)]
mod tests {
  use ndarray::arr2;

  use super::*;

  const N: usize = 10000;

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
}
