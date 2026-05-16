//! Finite-difference Malliavin sensitivity traits.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::traits::FloatExt;

use crate::noise::gn::Gn;
use crate::traits::process::ProcessExt;

pub trait MalliavinExt<T: FloatExt> {
  fn sample_with_noise(&self, noise: &Array1<T>) -> Array1<T>;

  fn n(&self) -> usize;

  fn t(&self) -> Option<T>;

  fn malliavin_derivative<F>(&self, f: F, epsilon: T) -> Array1<T>
  where
    F: Fn(&Array1<T>) -> T,
  {
    let gn = Gn::new(self.n() - 1, self.t(), Unseeded);
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

  fn malliavin_derivative_terminal(&self, epsilon: T) -> Array1<T> {
    self.malliavin_derivative(|path| *path.last().unwrap(), epsilon)
  }
}

pub trait Malliavin2DExt<T: FloatExt> {
  fn sample_with_noise(&self, noise: &[Array1<T>; 2]) -> [Array1<T>; 2];

  fn generate_noise(&self) -> [Array1<T>; 2];

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
