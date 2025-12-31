use std::f64::consts::PI;

use ndarray::Array1;

/// A struct representing a Gaussian Kernel Density Estimator (KDE).
///
/// # Fields
/// - `data`: 1D array of data points.
/// - `bandwidth`: The bandwidth (smoothing parameter) for the Gaussian kernel.
#[derive(Debug)]
pub struct GaussianKDE {
  data: Array1<f64>,
  bandwidth: f64,
}

impl GaussianKDE {
  /// Creates a new `GaussianKDE` with given data and bandwidth.
  ///
  /// # Arguments
  ///
  /// * `data` - 1D array of data points.
  /// * `bandwidth` - The smoothing parameter for the Gaussian kernel.
  ///
  /// # Returns
  ///
  /// A `GaussianKDE` instance containing the specified data and bandwidth.
  ///
  /// # Examples
  ///
  /// ```
  /// use ndarray::Array1;
  /// // Suppose we already have a GaussianKDE struct in scope
  /// // let data = Array1::from(vec![1.0, 2.0, 3.0]);
  /// // let kde = GaussianKDE::new(data, 0.5);
  /// ```
  pub fn new(data: Array1<f64>, bandwidth: f64) -> Self {
    Self { data, bandwidth }
  }

  /// Creates a new `GaussianKDE` where the bandwidth is automatically
  /// chosen by Silverman's rule of thumb.
  ///
  /// Silverman’s rule of thumb (for 1D) is:
  ///
  /// `h = 0.9 * min(σ, IQR/1.34) * n^(-1/5)`,
  ///
  /// where:
  /// - `σ` is the standard deviation of the data,
  /// - `IQR` is the interquartile range (75th percentile - 25th percentile),
  /// - `n` is the number of data points.
  ///
  /// # Arguments
  ///
  /// * `data` - 1D array of data points.
  ///
  /// # Returns
  ///
  /// A `GaussianKDE` instance with bandwidth estimated by Silverman's rule.
  pub fn with_silverman_bandwidth(data: Array1<f64>) -> Self {
    let h = silverman_bandwidth(&data);
    Self { data, bandwidth: h }
  }

  /// Evaluates the Gaussian kernel (normal PDF) for a given `x`, centered at `xi`,
  /// using the instance's bandwidth.
  ///
  /// # Arguments
  ///
  /// * `x`  - The point at which to evaluate the kernel.
  /// * `xi` - The center of the kernel (an individual data point).
  ///
  /// # Returns
  ///
  /// The kernel value at `x`.
  fn gaussian_kernel(&self, x: f64, xi: f64) -> f64 {
    let norm_factor = 1.0 / (self.bandwidth * (2.0 * PI).sqrt());
    let exponent = -0.5 * ((x - xi) / self.bandwidth).powi(2);
    norm_factor * exponent.exp()
  }

  /// Evaluates the Gaussian KDE at a single point `x`.
  ///
  /// # Arguments
  ///
  /// * `x` - The point at which to evaluate the KDE.
  ///
  /// # Returns
  ///
  /// The estimated density at `x`.
  ///
  /// # Examples
  ///
  /// ```
  /// // let kde = GaussianKDE::new(data, 0.5);
  /// // let density_value = kde.evaluate(1.5);
  /// ```
  pub fn evaluate(&self, x: f64) -> f64 {
    let sum: f64 = self
      .data
      .iter()
      .map(|&xi| self.gaussian_kernel(x, xi))
      .sum();
    sum / (self.data.len() as f64)
  }

  /// Evaluates the Gaussian KDE for multiple values of `x` (an array of points).
  ///
  /// # Arguments
  ///
  /// * `x_values` - 1D array of points at which to evaluate the KDE.
  ///
  /// # Returns
  ///
  /// A 1D array containing the density estimates for each point in `x_values`.
  ///
  /// # Examples
  ///
  /// ```
  /// // let kde = GaussianKDE::new(data, 0.5);
  /// // let xs = Array1::linspace(0.0, 5.0, 50);
  /// // let ys = kde.evaluate_array(&xs);
  /// ```
  pub fn evaluate_array(&self, x_values: &Array1<f64>) -> Array1<f64> {
    x_values.mapv(|x| self.evaluate(x))
  }
}

/// Computes the bandwidth using Silverman's rule of thumb for 1D data.
///
/// Silverman’s rule of thumb (for 1D) is:
///
/// `h = 0.9 * min(σ, IQR/1.34) * n^(-1/5)`,
///
/// - `σ` = standard deviation of the data,
/// - `IQR` = interquartile range = 75th percentile - 25th percentile,
/// - `n` = number of data points.
///
/// # Arguments
///
/// * `data` - 1D array of data points.
///
/// # Returns
///
/// The estimated bandwidth.
pub fn silverman_bandwidth(data: &Array1<f64>) -> f64 {
  let n = data.len() as f64;
  if n < 2.0 {
    // If there's only one data point or empty, fallback to something minimal.
    return 1e-6;
  }

  // Compute the standard deviation.
  let mean = data.mean().unwrap_or(0.0);
  let std = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();

  // Compute the interquartile range.
  let mut sorted = data.to_vec();
  sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
  let q1 = percentile(&sorted, 25.0);
  let q3 = percentile(&sorted, 75.0);
  let iqr = q3 - q1;

  let scale = std.min(iqr / 1.34);
  let h = 0.9 * scale * (n.powf(-1.0 / 5.0));
  // Avoid extremely small bandwidth.
  if h < 1e-8 {
    1e-8
  } else {
    h
  }
}

/// Returns the p-th percentile of a sorted vector.
///
/// # Arguments
///
/// * `sorted_data` - A sorted vector of floating-point values.
/// * `p`           - The percentile (0..100).
///
/// # Returns
///
/// The value corresponding to the p-th percentile.
pub fn percentile(sorted_data: &Vec<f64>, p: f64) -> f64 {
  if sorted_data.is_empty() {
    return 0.0;
  }
  if p <= 0.0 {
    return sorted_data[0];
  }
  if p >= 100.0 {
    return sorted_data[sorted_data.len() - 1];
  }

  let rank = (p / 100.0) * (sorted_data.len() as f64 - 1.0);
  let lower_index = rank.floor() as usize;
  let upper_index = rank.ceil() as usize;

  if lower_index == upper_index {
    sorted_data[lower_index]
  } else {
    // Linear interpolation between lower and upper index
    let weight = rank - lower_index as f64;
    let lower_val = sorted_data[lower_index];
    let upper_val = sorted_data[upper_index];
    lower_val + weight * (upper_val - lower_val)
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array;
  use ndarray::Array1;

  use super::*;

  #[test]
  fn test_kde() {
    let data = Array1::from(vec![1.0, 1.5, 2.0, 2.5, 3.0]);

    let kde_auto = GaussianKDE::with_silverman_bandwidth(data.clone());
    println!("Silverman bandwidth: {:.5}", kde_auto.bandwidth);

    let x_single = 2.0;
    let density_single = kde_auto.evaluate(x_single);
    println!("KDE({}) = {:.6}", x_single, density_single);

    let x_values = Array::linspace(0.0, 4.0, 11);
    let density_values = kde_auto.evaluate_array(&x_values);
    println!("\nEvaluating multiple points (0..4):");
    for (x, dens) in x_values.iter().zip(density_values.iter()) {
      println!("x = {:.2}, KDE = {:.6}", x, dens);
    }
  }

  #[test]
  fn test_kde_evaluate_single() {
    let data = Array1::from(vec![1.0, 2.0, 3.0]);
    let kde = GaussianKDE::new(data.clone(), 0.5);
    let density = kde.evaluate(2.0);

    assert!(density.is_finite());
    assert!(density >= 0.0);
  }

  #[test]
  fn test_silverman_bandwidth() {
    let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let h = silverman_bandwidth(&data);

    assert!(h > 0.0, "Bandwidth should be positive");
    assert!(h < 10.0, "Bandwidth seems unexpectedly large");
  }
}
