//! # Samples
//!
//! $$
//! F_{X_1,\dots,X_d}(x)=C\left(F_1(x_1),\dots,F_d(x_d)\right)
//! $$
//!
use ndarray::Array1;
use ndarray::Array2;
use plotly::Plot;
use plotly::Scatter;
use rand_distr::Distribution;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

use crate::distributions::exp::SimdExp;
use crate::distributions::gamma::SimdGamma;
use crate::distributions::normal::SimdNormal;
pub use crate::traits::NCopula2DExt;

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
  (-s.powf(1.0 / theta)).exp()
}

/// Gaussian copula (2D)
#[derive(Clone, Debug)]
pub struct GaussianCopula2D {
  /// 2D mean vector, e.g. [0.0, 0.0]
  pub mean: Array1<f64>,
  /// 2x2 covariance matrix
  pub cov: Array2<f64>,
}

impl NCopula2DExt for GaussianCopula2D {
  fn sample(&self, n: usize) -> Array2<f64> {
    let a11 = self.cov[[0, 0]].sqrt();
    let a21 = self.cov[[1, 0]] / a11;
    let a22 = (self.cov[[1, 1]] - a21 * a21).sqrt();

    let n1: SimdNormal<f64> = SimdNormal::new(0.0, 1.0);
    let n2: SimdNormal<f64> = SimdNormal::new(0.0, 1.0);

    let mut rng = rand::rng();
    let mut z = Array2::<f64>::zeros((n, 2));

    for i in 0..n {
      let z1: f64 = n1.sample(&mut rng);
      let z2: f64 = n2.sample(&mut rng);
      z[[i, 0]] = self.mean[0] + a11 * z1;
      z[[i, 1]] = self.mean[1] + a21 * z1 + a22 * z2;
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

impl NCopula2DExt for GumbelCopula2D {
  fn sample(&self, n: usize) -> Array2<f64> {
    let alpha = self.alpha;
    assert!(alpha >= 1.0, "The Gumbel parameter (alpha) must be >= 1!");

    let mut rng = rand::rng();
    let exp_dist: SimdExp<f64> = SimdExp::new(1.0);
    let mut data = Array2::<f64>::zeros((n, 2));

    for i in 0..n {
      let x: f64 = exp_dist.sample(&mut rng);
      let m = x.powf(alpha);

      let e1: f64 = exp_dist.sample(&mut rng);
      let e2: f64 = exp_dist.sample(&mut rng);

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

impl NCopula2DExt for ClaytonCopula2D {
  fn sample(&self, n: usize) -> Array2<f64> {
    let alpha = self.alpha;
    assert!(alpha > 0.0, "The Clayton parameter (alpha) must be > 0!");

    let mut rng = rand::rng();
    let gamma_dist: SimdGamma<f64> = SimdGamma::new(1.0 / alpha, 1.0);
    let exp_dist: SimdExp<f64> = SimdExp::new(1.0);

    let mut data = Array2::<f64>::zeros((n, 2));

    for i in 0..n {
      let w: f64 = gamma_dist.sample(&mut rng);

      let e1: f64 = exp_dist.sample(&mut rng);
      let e2: f64 = exp_dist.sample(&mut rng);

      let u1: f64 = (1.0 + w * e1).powf(-1.0 / alpha);
      let u2: f64 = (1.0 + w * e2).powf(-1.0 / alpha);

      data[[i, 0]] = u1;
      data[[i, 1]] = u2;
    }
    data
  }

  fn get_params(&self) -> Vec<f64> {
    vec![self.alpha]
  }
}
