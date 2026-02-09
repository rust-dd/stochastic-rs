use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct Jacobi<T: Float> {
  pub alpha: T,
  pub beta: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
}

impl<T: Float> Jacobi<T> {
  fn new(alpha: T, beta: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    assert!(alpha > T::zero(), "alpha must be positive");
    assert!(beta > T::zero(), "beta must be positive");
    assert!(sigma > T::zero(), "sigma must be positive");
    assert!(alpha < beta, "alpha must be less than beta");

    Jacobi {
      alpha,
      beta,
      sigma,
      n,
      x0,
      t,
    }
  }
}

impl<T: Float> Process<T> for Jacobi<T> {
  type Output = Array1<T>;
  type Noise = Gn<T>;

  /// Sample the Jacobi process
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

    let mut jacobi = Array1::<T>::zeros(self.n);
    jacobi[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      jacobi[i] = match jacobi[i - 1] {
        _ if jacobi[i - 1] <= T::zero() && i > 0 => T::zero(),
        _ if jacobi[i - 1] >= T::one() && i > 0 => T::one(),
        _ => {
          jacobi[i - 1]
            + (self.alpha - self.beta * jacobi[i - 1]) * dt
            + self.sigma * (jacobi[i - 1] * (T::one() - jacobi[i - 1])).sqrt() * gn[i - 1]
        }
      }
    }

    jacobi
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::plot_1d;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn fjacobi_length_equals_n() {
    let jacobi = Jacobi::new(0.43, 0.5, 0.8, N, Some(X0), Some(1.0));
    assert_eq!(jacobi.sample().len(), N);
  }

  #[test]
  fn jacobi_starts_with_x0() {
    let jacobi = Jacobi::new(0.43, 0.5, 0.8, N, Some(X0), Some(1.0));
    assert_eq!(jacobi.sample()[0], X0);
  }

  #[test]
  fn jacobi_plot() {
    let jacobi = Jacobi::new(0.43, 0.5, 0.8, N, Some(X0), Some(1.0));
    plot_1d!(jacobi.sample(), "Jacobi process");
  }

  #[test]
  #[ignore = "Not implemented"]
  fn fjacobi_malliavin() {
    unimplemented!();
  }
}
