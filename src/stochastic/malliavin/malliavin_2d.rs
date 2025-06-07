use ndarray::Array1;

use crate::stochastic::Sampling2DExt;

/// This module provides tools for Malliavin calculus.
/// The implementation is based on the perturbation method and it is very simple and still unstable.
/// Use with caution.
pub trait Malliavin2D {
  /// Compute the Malliavin derivative of the stochastic process using perturbation.
  ///
  /// - `f`: the function that maps the path to a scalar value.
  /// - `epsilon`: the perturbation value.
  ///
  /// The Malliavin derivative is defined as the derivative of the function `f` with respect to the path.
  fn malliavin_derivate<F>(&self, f: F, epsilon: f64, idx: Option<usize>) -> Array1<f64>
  where
    F: Fn(&Array1<f64>) -> f64,
  {
    let mut path = self.path().get(idx.unwrap_or(0)).unwrap().to_owned();
    let mut derivates = Array1::zeros(path.len() + 1);
    let f_original = f(&path);

    for i in 1..=self.path().len() {
      let original_value = path[i];
      path[i] += epsilon;
      let f_perturbed = f(&path);
      derivates[i] = (f_perturbed - f_original) / epsilon;
      path[i] = original_value;
    }

    derivates
  }

  /// Compute the Malliavin derivative of the stochastic process using perturbation for latest value.
  ///
  ///  - `epsilon`: the perturbation value.
  ///
  /// The Malliavin derivative is defined as the derivative of the function `f` with respect to the path.
  /// For example we want to know how the option price changes if the stock price changes.
  fn malliavin_derivate_latest(&self, epsilon: f64, idx: Option<usize>) -> Array1<f64> {
    let mut path = self.path().get(idx.unwrap_or(0)).unwrap().to_owned();
    let mut derivates = Array1::zeros(path.len() + 1);

    let final_value = |path: &Array1<f64>| -> f64 { *path.last().unwrap() };
    let f_original = final_value(&path);

    for i in 1..=path.len() {
      let original_value = path[i];
      path[i] += epsilon;
      let f_perturbed = final_value(&path);
      derivates[i] = (f_perturbed - f_original) / epsilon;
      path[i] = original_value;
    }

    derivates
  }

  /// Get stochastic process path.
  fn path(&self) -> [Array1<f64>; 2];
}

impl<T: Sampling2DExt<f64>> Malliavin2D for T {
  fn path(&self) -> [Array1<f64>; 2] {
    self.sample()
  }
}
