use ndarray::Array1;

use crate::stochastic::c;
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
  gn: Gn<T>,
}

impl<T: Float> CIR<T> {
  /// Create a new CIR process.
  pub fn new(
    theta: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
  ) -> Self {
    assert!(
      c::<T>(2.0) * theta * mu >= sigma.powi(2),
      "2 * theta * mu < sigma^2"
    );

    Self {
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      use_sym,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: Float> Process<T> for CIR<T> {
  type Output = Array1<T>;

  /// Sample the Cox-Ingersoll-Ross (CIR) process
  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = &self.gn.sample();

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
    let cir = CIR::new(1.0, 1.2, 0.2, N, Some(X0), Some(1.0), Some(false));
    assert_eq!(cir.sample().len(), N);
  }

  #[test]
  fn cir_starts_with_x0() {
    let cir = CIR::new(1.0, 1.2, 0.2, N, Some(X0), Some(1.0), Some(false));
    assert_eq!(cir.sample()[0], X0);
  }

  #[test]
  fn cir_plot() {
    let cir = CIR::new(1.0, 1.2, 0.2, N, Some(X0), Some(1.0), Some(false));
    plot_1d!(cir.sample(), "Cox-Ingersoll-Ross (CIR) process");
  }

  #[test]
  #[ignore = "Not implemented"]
  fn cir_malliavin() {
    unimplemented!();
  }
}
