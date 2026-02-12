use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct Jacobi<T: FloatExt> {
  pub alpha: T,
  pub beta: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  gn: Gn<T>,
}

impl<T: FloatExt> Jacobi<T> {
  pub fn new(alpha: T, beta: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
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
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for Jacobi<T> {
  type Output = Array1<T>;

  /// Sample the Jacobi process
  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = &self.gn.sample();

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
