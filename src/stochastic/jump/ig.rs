use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct IG<T: Float> {
  pub gamma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  gn: Gn<T>,
}

impl<T: Float> IG<T> {
  pub fn new(gamma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      gamma,
      n,
      x0,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: Float> Process<T> for IG<T> {
  type Output = Array1<T>;
  type Noise = Gn<T>;

  fn sample(&self) -> Self::Output {
    self.euler_maruyama(|gn| gn.sample())
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Self::Output {
    self.euler_maruyama(|gn| gn.sample_simd())
  }

  fn euler_maruyama(
    &self,
    noise_fn: impl Fn(&Self::Noise) -> <Self::Noise as Process<T>>::Output,
  ) -> Self::Output {
    let dt = self.gn.dt();
    let gn = noise_fn(&self.gn);
    let mut ig = Array1::zeros(self.n);
    ig[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      ig[i] = ig[i - 1] + self.gamma * dt + gn[i - 1]
    }

    ig
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::plot_1d;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn ig_length_equals_n() {
    let ig = IG::new(2.25, N, Some(X0), Some(10.0));
    assert_eq!(ig.sample().len(), N);
  }

  #[test]
  fn ig_starts_with_x0() {
    let ig = IG::new(2.25, N, Some(X0), Some(10.0));
    assert_eq!(ig.sample()[0], X0);
  }

  #[test]
  fn ig_plot() {
    let ig = IG::new(2.25, N, Some(X0), Some(10.0));
    plot_1d!(ig.sample(), "Inverse Gaussian (IG)");
  }
}
