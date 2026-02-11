use ndarray::Array1;

use crate::stochastic::noise::fgn::FGN;
use crate::stochastic::Float;
use crate::stochastic::Process;

/// Fractional Cox-Ingersoll-Ross (FCIR) process.
/// dX(t) = theta(mu - X(t))dt + sigma * sqrt(X(t))dW^H(t)
/// where X(t) is the FCIR process.
pub struct FCIR<T: Float> {
  pub hurst: T,
  pub theta: T,
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub use_sym: Option<bool>,
  fgn: FGN<T>,
}

impl<T: Float> FCIR<T> {
  #[must_use]
  pub fn new(
    hurst: T,
    theta: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
  ) -> Self {
    assert!(
      T::from_usize_(2) * theta * mu >= sigma.powi(2),
      "2 * theta * mu < sigma^2"
    );

    Self {
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      use_sym,
      fgn: FGN::new(hurst, n - 1, t),
    }
  }
}

impl<T: Float> Process<T> for FCIR<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.fgn.dt();
    let fgn = &self.fgn.sample();

    let mut fcir = Array1::<T>::zeros(self.n);
    fcir[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      let dfcir = self.theta * (self.mu - fcir[i - 1]) * dt
        + self.sigma * (fcir[i - 1]).abs().sqrt() * fgn[i - 1];

      fcir[i] = match self.use_sym.unwrap_or(false) {
        true => (fcir[i - 1] + dfcir).abs(),
        false => (fcir[i - 1] + dfcir).max(T::zero()),
      };
    }

    fcir
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::plot_1d;
  use crate::stochastic::N;
  use crate::stochastic::X0;

  #[test]
  fn fcir_length_equals_n() {
    let fcir = FCIR::<f64>::new(0.7, 1.0, 1.2, 0.2, N, Some(X0), Some(1.0), Some(false));

    assert_eq!(fcir.sample().len(), N);
  }

  #[test]
  fn fcir_starts_with_x0() {
    let fcir = FCIR::<f64>::new(0.7, 1.0, 1.2, 0.2, N, Some(X0), Some(1.0), Some(false));

    assert_eq!(fcir.sample()[0], X0);
  }

  #[test]
  fn fcir_plot() {
    let fcir = FCIR::<f64>::new(0.7, 1.0, 1.2, 0.2, N, Some(X0), Some(1.0), Some(false));

    plot_1d!(
      fcir.sample(),
      "Fractional Cox-Ingersoll-Ross (FCIR) process"
    );
  }
}
