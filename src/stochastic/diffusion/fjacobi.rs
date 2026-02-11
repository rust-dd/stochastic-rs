use ndarray::Array1;

use crate::stochastic::noise::fgn::FGN;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct FJacobi<T: Float> {
  pub hurst: T,
  pub alpha: T,
  pub beta: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  fgn: FGN<T>,
}

impl<T: Float> FJacobi<T> {
  #[must_use]
  pub fn new(hurst: T, alpha: T, beta: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    assert!(alpha > T::zero(), "alpha must be positive");
    assert!(beta > T::zero(), "beta must be positive");
    assert!(sigma > T::zero(), "sigma must be positive");
    assert!(alpha < beta, "alpha must be less than beta");

    Self {
      hurst,
      alpha,
      beta,
      sigma,
      n,
      x0,
      t,
      fgn: FGN::new(hurst, n - 1, t),
    }
  }
}

impl<T: Float> Process<T> for FJacobi<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.fgn.dt();
    let fgn = self.fgn.sample();

    let mut fjacobi = Array1::<T>::zeros(self.n);
    fjacobi[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      fjacobi[i] = match fjacobi[i - 1] {
        _ if fjacobi[i - 1] <= T::zero() && i > 0 => T::zero(),
        _ if fjacobi[i - 1] >= T::one() && i > 0 => T::one(),
        _ => {
          fjacobi[i - 1]
            + (self.alpha - self.beta * fjacobi[i - 1]) * dt
            + self.sigma * (fjacobi[i - 1] * (T::one() - fjacobi[i - 1])).sqrt() * fgn[i - 1]
        }
      };
    }

    fjacobi
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
    let fjacobi = FJacobi::<f64>::new(0.7, 0.43, 0.5, 0.8, N, Some(X0), Some(1.0));

    assert_eq!(fjacobi.sample().len(), N);
  }

  #[test]
  fn fjacobi_starts_with_x0() {
    let fjacobi = FJacobi::<f64>::new(0.7, 0.43, 0.5, 0.8, N, Some(X0), Some(1.0));

    assert_eq!(fjacobi.sample()[0], X0);
  }

  #[test]
  fn fjacobi_plot() {
    let fjacobi = FJacobi::<f64>::new(0.7, 0.43, 0.5, 0.8, N, Some(X0), Some(1.0));

    plot_1d!(fjacobi.sample(), "Fractional Jacobi process");
  }
}
