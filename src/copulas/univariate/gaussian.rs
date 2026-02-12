use ndarray::Array1;
use statrs::distribution::Continuous;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

#[derive(Debug, Clone)]
pub struct GaussianUnivariate {
  dist: Option<Normal>, // statrs Normal distribution
}

impl GaussianUnivariate {
  pub fn new() -> Self {
    Self { dist: None }
  }

  /// Fit mean and std to data column, then store a statrs Normal distribution.
  pub fn fit(&mut self, column: &Array1<f64>) {
    let n = column.len() as f64;
    if n < 2.0 {
      // fallback: something trivial
      self.dist = Some(Normal::new(0.0, 1.0).unwrap());
      return;
    }
    let mean = column.mean().unwrap_or(0.0);
    let var = column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = var.sqrt().max(1e-12);

    // Create a Normal distribution
    self.dist = Some(Normal::new(mean, std).unwrap());
  }

  pub fn is_fitted(&self) -> bool {
    self.dist.is_some()
  }

  /// Shortcut for the underlying normal CDF.
  pub fn cdf(&self, x: f64) -> f64 {
    if let Some(d) = &self.dist {
      d.cdf(x)
    } else {
      0.5
    }
  }

  /// Shortcut for the underlying normal PDF.
  pub fn pdf(&self, x: f64) -> f64 {
    if let Some(d) = &self.dist {
      d.pdf(x)
    } else {
      1.0
    }
  }

  /// Shortcut for the underlying normal PPF (inverse CDF).
  pub fn ppf(&self, p: f64) -> f64 {
    if let Some(d) = &self.dist {
      d.inverse_cdf(p)
    } else {
      0.0
    }
  }
}
