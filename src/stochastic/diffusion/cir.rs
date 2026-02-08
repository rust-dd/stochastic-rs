use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

/// Cox-Ingersoll-Ross (CIR) process.
/// dX(t) = theta(mu - X(t))dt + sigma * sqrt(X(t))dW(t)
/// where X(t) is the CIR process.
pub struct CIR<T: Float> {
  pub theta: T,
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub use_sym: Option<bool>,
  pub m: Option<usize>,
}

impl<T: Float> Process<T> for CIR<T> {
  type Output = Array1<T>;

  /// Sample the Cox-Ingersoll-Ross (CIR) process
  fn sample(&self) -> Self::Output {
    assert!(
      2.0 * self.theta * self.mu >= self.sigma.powi(2),
      "2 * theta * mu < sigma^2"
    );

    let gn = Gn::new(self.n - 1, self.t);
    let dt = gn.dt();
    let gn = if cfg!(feature = "simd") {
      gn.sample_simd()
    } else {
      gn.sample()
    };

    let mut cir = Array1::<T>::zeros(self.n);
    cir[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      let dcir = self.theta * (self.mu - cir[i - 1]) * dt
        + self.sigma * (cir[i - 1]).abs().sqrt() * gn[i - 1];

      cir[i] = match self.use_sym.unwrap_or(false) {
        true => (cir[i - 1] + dcir).abs(),
        false => (cir[i - 1] + dcir).max(T::zero()),
      };
    }

    cir
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Self::Output {
    assert!(
      2.0 * self.theta * self.mu >= self.sigma.powi(2),
      "2 * theta * mu < sigma^2"
    );

    let gn = Gn::new(self.n - 1, self.t);
    let dt = gn.dt();
    let gn = gn.sample_simd();

    let mut cir = Array1::<f32>::zeros(self.n);
    cir[0] = self.x0.unwrap_or(0.0);

    for i in 1..self.n {
      let dcir = self.theta * (self.mu - cir[i - 1]) * dt
        + self.sigma * (cir[i - 1]).abs().sqrt() * gn[i - 1];

      cir[i] = match self.use_sym.unwrap_or(false) {
        true => (cir[i - 1] + dcir).abs(),
        false => (cir[i - 1] + dcir).max(0.0),
      };
    }

    cir
  }

  /// Number of time steps
  fn n(&self) -> usize {
    self.n
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::plot_1d;
  use crate::stochastic::Process;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn cir_length_equals_n() {
    let cir = CIR::new(1.0, 1.2, 0.2, N, Some(X0), Some(1.0), Some(false), None);
    assert_eq!(cir.sample().len(), N);
  }

  #[test]
  fn cir_starts_with_x0() {
    let cir = CIR::new(1.0, 1.2, 0.2, N, Some(X0), Some(1.0), Some(false), None);
    assert_eq!(cir.sample()[0], X0);
  }

  #[test]
  fn cir_plot() {
    let cir = CIR::new(1.0, 1.2, 0.2, N, Some(X0), Some(1.0), Some(false), None);
    plot_1d!(cir.sample(), "Cox-Ingersoll-Ross (CIR) process");
  }

  #[test]
  #[ignore = "Not implemented"]
  fn cir_malliavin() {
    unimplemented!();
  }
}
