use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub trait Malliavin<T: Float> {
  /// Build a path from given noise increments.
  fn sample_with_noise(&self, noise: &Array1<T>) -> Array1<T>;

  fn n(&self) -> usize;

  fn t(&self) -> Option<T>;

  /// Compute the Malliavin derivative using noise perturbation.
  /// Perturbs each dW_i by epsilon, rebuilds the path, and computes the finite difference.
  fn malliavin_derivative<F>(&self, f: F, epsilon: T) -> Array1<T>
  where
    F: Fn(&Array1<T>) -> T,
  {
    let gn = Gn::new(self.n() - 1, self.t());
    let mut noise = gn.sample();
    let path = self.sample_with_noise(&noise);
    let f_original = f(&path);
    let mut derivatives = Array1::zeros(noise.len());

    for i in 0..noise.len() {
      let original = noise[i];
      noise[i] += epsilon;
      let path_perturbed = self.sample_with_noise(&noise);
      derivatives[i] = (f(&path_perturbed) - f_original) / epsilon;
      noise[i] = original;
    }

    derivatives
  }

  /// Malliavin derivative of the terminal value.
  fn malliavin_derivative_terminal(&self, epsilon: T) -> Array1<T> {
    self.malliavin_derivative(|path| *path.last().unwrap(), epsilon)
  }
}
