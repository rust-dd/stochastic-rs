use ndarray::Array1;

use crate::stochastic::FloatExt;

pub trait Malliavin2DExt<T: FloatExt> {
  /// Build paths from given noise increments.
  fn sample_with_noise(&self, noise: &[Array1<T>; 2]) -> [Array1<T>; 2];

  /// Generate noise increments (independent Brownian increments).
  fn generate_noise(&self) -> [Array1<T>; 2];

  /// Compute the Malliavin derivative by perturbing a noise component.
  /// `noise_component` selects which driving Brownian motion to perturb (0 or 1).
  fn malliavin_derivative<F>(&self, f: F, epsilon: T, noise_component: usize) -> Array1<T>
  where
    F: Fn(&[Array1<T>; 2]) -> T,
  {
    let mut noise = self.generate_noise();
    let paths = self.sample_with_noise(&noise);
    let f_original = f(&paths);
    let n = noise[noise_component].len();
    let mut derivatives = Array1::zeros(n);

    for i in 0..n {
      let original = noise[noise_component][i];
      noise[noise_component][i] += epsilon;
      let paths_perturbed = self.sample_with_noise(&noise);
      derivatives[i] = (f(&paths_perturbed) - f_original) / epsilon;
      noise[noise_component][i] = original;
    }

    derivatives
  }

  /// Malliavin derivative of the terminal value of `path_component`
  /// with respect to `noise_component`.
  fn malliavin_derivative_terminal(
    &self,
    epsilon: T,
    path_component: usize,
    noise_component: usize,
  ) -> Array1<T> {
    self.malliavin_derivative(
      |paths| *paths[path_component].last().unwrap(),
      epsilon,
      noise_component,
    )
  }
}
