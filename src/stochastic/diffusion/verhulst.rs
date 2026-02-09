use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

/// Verhulst (logistic) diffusion
/// dX_t = r X_t (1 - X_t / K) dt + sigma X_t dW_t
pub struct Verhulst<T: Float> {
  pub r: T,
  pub k: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  /// If true, clamp the state into [0, K] each step
  pub clamp: Option<bool>,
}

impl<T: Float> Verhulst<T> {
  pub fn new(
    r: T,
    k: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    clamp: Option<bool>,
  ) -> Self {
    Self {
      r,
      k,
      sigma,
      n,
      x0,
      t,
      clamp,
    }
  }
}

impl<T: Float> Process<T> for Verhulst<T> {
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
      let drift = self.r * xi * (1.0 - xi / self.k) * dt;
      let diff = self.sigma * xi * gn[i - 1];
      let mut next = xi + drift + diff;
      if self.clamp.unwrap_or(true) {
        next = next.clamp(T::zero(), self.k);
      }
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
  fn verhulst_length_equals_n() {
    let proc = Verhulst::new(1.2, 1.0, 0.3, N, Some(X0), Some(1.0), Some(true));
    assert_eq!(proc.sample().len(), N);
  }

  #[test]
  fn verhulst_starts_with_x0() {
    let proc = Verhulst::new(1.2, 1.0, 0.3, N, Some(X0), Some(1.0), Some(true));
    assert_eq!(proc.sample()[0], X0);
  }

  #[test]
  fn verhulst_plot() {
    let proc = Verhulst::new(1.2, 1.0, 0.3, N, Some(X0), Some(1.0), Some(true));
    plot_1d!(proc.sample(), "Verhulst (logistic) diffusion");
  }
}
