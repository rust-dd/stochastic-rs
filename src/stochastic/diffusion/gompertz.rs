use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

/// Gompertz diffusion
/// dX_t = (a - b ln X_t) X_t dt + sigma X_t dW_t
pub struct Gompertz<T: Float> {
  pub a: T,
  pub b: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
}

impl<T: Float> Gompertz<T> {
  pub fn new(a: T, b: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      a,
      b,
      sigma,
      n,
      x0,
      t,
    }
  }
}

impl<T: Float> Process<T> for Gompertz<T> {
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
    let gn = noise_fn(&gn);

    let mut x = Array1::<T>::zeros(self.n);
    x[0] = self.x0.unwrap_or(T::zero()).max(1e-12.into());

    for i in 1..self.n {
      let xi = x[i - 1].max(1e-12.into());
      let drift = (self.a - self.b * xi.ln()) * xi * dt;
      let diff = self.sigma * xi * gn[i - 1];
      let next = xi + drift + diff;
      x[i] = next.max(1e-12.into());
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
  fn gompertz_length_equals_n() {
    let proc = Gompertz::new(1.0, 0.5, 0.3, N, Some(X0), Some(1.0));
    assert_eq!(proc.sample().len(), N);
  }

  #[test]
  fn gompertz_starts_with_x0() {
    let proc = Gompertz::new(1.0, 0.5, 0.3, N, Some(X0), Some(1.0));
    assert_eq!(proc.sample()[0], X0.max(1e-12));
  }

  #[test]
  fn gompertz_plot() {
    let proc = Gompertz::new(1.0, 0.5, 0.3, N, Some(X0), Some(1.0));
    plot_1d!(proc.sample(), "Gompertz diffusion");
  }
}
