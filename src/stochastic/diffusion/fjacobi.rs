use ndarray::Array1;

use crate::stochastic::noise::fgn::FGN;
use crate::stochastic::FloatExt;
use crate::stochastic::ProcessExt;

pub struct FJacobi<T: FloatExt> {
  pub hurst: T,
  pub alpha: T,
  pub beta: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  fgn: FGN<T>,
}

impl<T: FloatExt> FJacobi<T> {
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

impl<T: FloatExt> ProcessExt<T> for FJacobi<T> {
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
