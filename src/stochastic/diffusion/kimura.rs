use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

/// Kimura / Wright–Fisher diffusion
/// dX_t = a X_t (1 - X_t) dt + sigma sqrt(X_t (1 - X_t)) dW_t
pub struct Kimura<T: Float> {
  pub a: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub gn: Gn<T>,
}

impl<T: Float> Kimura<T> {
  pub fn new(a: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Kimura {
      a,
      sigma,
      n,
      x0,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: Float> Process<T> for Kimura<T> {
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

    let mut x = Array1::<T>::zeros(self.n);
    x[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      // enforce [0,1] domain when computing coefficients
      let xi = x[i - 1].clamp(T::zero(), T::one());
      let sqrt_term = (xi * (1.0 - xi)).sqrt();
      let drift = self.a * xi * (1.0 - xi) * dt;
      let diff = self.sigma * sqrt_term * gn[i - 1];
      let mut next = xi + drift + diff;
      next = next.clamp(T::zero(), T::one());
      x[i] = next;
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
  fn kimura_length_equals_n() {
    let proc = Kimura::new(2.0, 0.5, N, Some(X0), Some(1.0));
    assert_eq!(proc.sample().len(), N);
  }

  #[test]
  fn kimura_starts_with_x0() {
    let proc = Kimura::new(2.0, 0.5, N, Some(X0), Some(1.0));
    assert_eq!(proc.sample()[0], X0);
  }

  #[test]
  fn kimura_plot() {
    let proc = Kimura::new(2.0, 0.5, N, Some(X0), Some(1.0));
    plot_1d!(proc.sample(), "Kimura / Wright–Fisher diffusion");
  }
}
