use ndarray::Array1;

use crate::stochastic::Float;
use crate::stochastic::Process;

/// This module provides tools for Malliavin calculus.
/// The implementation is based on the perturbation method and it is very simple and still unstable.
/// Use with caution.
pub trait Malliavin<T: Float> {
  /// Compute the Malliavin derivative of the stochastic process using perturbation.
  ///
  /// - `f`: the function that maps the path to a scalar value.
  /// - `epsilon`: the perturbation value.
  ///
  /// The Malliavin derivative is defined as the derivative of the function `f` with respect to the path.
  fn malliavin_derivate<F>(&self, f: F, epsilon: T) -> Array1<T>
  where
    F: Fn(&Array1<T>) -> T,
  {
    let mut path = self.path();
    let mut derivates = Array1::zeros(path.len());
    let f_original = f(&path);

    for i in 1..path.len() {
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
  fn malliavin_derivate_latest(&self, epsilon: T) -> Array1<T> {
    let mut path = self.path();
    let mut derivates = Array1::zeros(path.len());

    let final_value = |path| -> T { *path.last().unwrap() };
    let f_original = final_value(&path);

    for i in 1..path.len() {
      let original_value = path[i];
      path[i] += epsilon;
      let f_perturbed = final_value(&path);
      derivates[i] = (f_perturbed - f_original) / epsilon;
      path[i] = original_value;
    }

    derivates
  }

  /// Get stochastic process path.
  fn path(&self) -> Array1<T>;
}

impl<T1: Float, T2: Process<T1, Output = Array1<T1>>> Malliavin<T1> for T2 {
  fn path(&self) -> Array1<T1> {
    self.sample()
  }
}
