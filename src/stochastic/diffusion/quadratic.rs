use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

/// Quadratic diffusion
/// dX_t = (alpha + beta X_t + gamma X_t^2) dt + sigma X_t dW_t
pub struct Quadratic<T: Float> {
  pub alpha: T,
  pub beta: T,
  pub gamma: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
}

impl<T: Float> Quadratic<T> {
  pub fn new(alpha: T, beta: T, gamma: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Quadratic {
      alpha,
      beta,
      gamma,
      sigma,
      n,
      x0,
      t,
    }
  }
}

impl<T: Float> Process<T> for Quadratic<T> {
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
    let gn = Gn::new(self.n - 1, self.t);
    let dt = gn.dt();
    let noise = noise_fn(&gn);

    let mut x = Array1::<T>::zeros(self.n);
    x[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      let xi = x[i - 1];
      let drift = (self.alpha + self.beta * xi + self.gamma * xi * xi) * dt;
      let diff = self.sigma * xi * gn[i - 1];
      x[i] = xi + drift + diff;
    }

    x
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::plot_1d;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn quadratic_length_equals_n() {
    let proc = Quadratic::new(0.1, 0.2, 0.1, 0.3, N, Some(X0), Some(1.0));
    assert_eq!(proc.sample().len(), N);
  }

  #[test]
  fn quadratic_starts_with_x0() {
    let proc = Quadratic::new(0.1, 0.2, 0.1, 0.3, N, Some(X0), Some(1.0));
    assert_eq!(proc.sample()[0], X0);
  }

  #[test]
  fn quadratic_plot() {
    let proc = Quadratic::new(0.1, 0.2, 0.1, 0.3, N, Some(X0), Some(1.0));
    plot_1d!(proc.sample(), "Quadratic diffusion");
  }
}
