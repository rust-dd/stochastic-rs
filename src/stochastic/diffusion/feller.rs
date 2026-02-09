use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

/// Feller–logistic diffusion
/// dX_t = kappa (theta - X_t) X_t dt + sigma sqrt(X_t) dW_t
pub struct FellerLogistic<T: Float> {
  pub kappa: T,
  pub theta: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  /// If true, reflect at 0; otherwise clamp at 0
  pub use_sym: Option<bool>,
}

impl<T: Float> FellerLogistic<T> {
  /// Create a new Feller–logistic diffusion process
  pub fn new(
    kappa: T,
    theta: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
  ) -> Self {
    Self {
      kappa,
      theta,
      sigma,
      n,
      x0,
      t,
      use_sym,
    }
  }
}

impl<T: Float> Process<T> for FellerLogistic<T> {
  type Output = Array1<T>;
  type Noise = Gn<T>;

  fn sample(&self) -> Self::Output {
    self.euler_maruyama(|gn| gn.sample())
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Self::Output {
    self.euler_maruyama(|gn| gn.sample_simd())
  }

  fn euler_maruyama(&self, noise_fn: impl FnOnce(&Self::Noise) -> Self::Output) -> Self::Output {
    let gn = Gn::new(self.n - 1, self.t);
    let dt = gn.dt();
    let gn = noise_fn(&gn);

    let mut x = Array1::<T>::zeros(self.n);
    x[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      let xi = x[i - 1].max(T::zero());
      let drift = self.kappa * (self.theta - xi) * xi * dt;
      let diff = self.sigma * xi.sqrt() * gn[i - 1];
      let next = xi + drift + diff;
      x[i] = match self.use_sym.unwrap_or(false) {
        true => next.abs(),
        false => next.max(T::zero()),
      };
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
  fn feller_length_equals_n() {
    let proc = FellerLogistic::new(1.0, 1.0, 0.2, N, Some(X0), Some(1.0), Some(false));
    assert_eq!(proc.sample().len(), N);
  }

  #[test]
  fn feller_starts_with_x0() {
    let proc = FellerLogistic::new(1.0, 1.0, 0.2, N, Some(X0), Some(1.0), Some(false));
    assert_eq!(proc.sample()[0], X0);
  }

  #[test]
  fn feller_plot() {
    let proc = FellerLogistic::new(1.0, 1.0, 0.2, N, Some(X0), Some(1.0), Some(false));
    plot_1d!(proc.sample(), "Feller–logistic diffusion");
  }
}
