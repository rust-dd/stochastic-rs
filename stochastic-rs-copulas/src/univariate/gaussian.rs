//! # Gaussian
//!
//! $$
//! C_\Sigma(u)=\Phi_\Sigma\!\left(\Phi^{-1}(u_1),\dots,\Phi^{-1}(u_d)\right)
//! $$
//!
use ndarray::Array1;
use stochastic_rs_distributions::special::ndtri;
use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_distributions::special::norm_pdf;

#[derive(Debug, Clone, Default)]
pub struct GaussianUnivariate {
  /// `Some((mean, std))` once `fit` has been called; `None` otherwise.
  params: Option<(f64, f64)>,
}

impl GaussianUnivariate {
  pub fn new() -> Self {
    Self::default()
  }

  /// Fit mean and std to data column, then cache the closed-form parameters.
  pub fn fit(&mut self, column: &Array1<f64>) {
    let n = column.len() as f64;
    if n < 2.0 {
      // fallback: standard normal
      self.params = Some((0.0, 1.0));
      return;
    }
    let mean = column.mean().unwrap_or(0.0);
    let var = column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = var.sqrt().max(1e-12);
    self.params = Some((mean, std));
  }

  pub fn is_fitted(&self) -> bool {
    self.params.is_some()
  }

  /// Normal CDF using the cached `(mean, std)`.
  pub fn cdf(&self, x: f64) -> f64 {
    match self.params {
      Some((mu, sigma)) => norm_cdf((x - mu) / sigma),
      None => 0.5,
    }
  }

  /// Normal pdf using the cached `(mean, std)`.
  pub fn pdf(&self, x: f64) -> f64 {
    match self.params {
      Some((mu, sigma)) => norm_pdf((x - mu) / sigma) / sigma,
      None => 1.0,
    }
  }

  /// Normal quantile (inverse CDF) using the cached `(mean, std)`.
  pub fn ppf(&self, p: f64) -> f64 {
    match self.params {
      Some((mu, sigma)) => mu + sigma * ndtri(p),
      None => 0.0,
    }
  }
}
