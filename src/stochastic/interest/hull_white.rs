use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::ProcessExt;

/// Hull-White process.
/// dX(t) = theta(t)dt - alpha * X(t)dt + sigma * dW(t)
/// where X(t) is the Hull-White process.
pub struct HullWhite<T: Float> {
  pub theta: fn(T) -> T,
  pub alpha: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  gn: Gn<T>,
}

impl<T: Float> HullWhite<T> {
  pub fn new(theta: fn(T) -> T, alpha: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      theta,
      alpha,
      sigma,
      n,
      x0,
      t,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: Float> ProcessExt<T> for HullWhite<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = &self.gn.sample();

    let mut hw = Array1::<T>::zeros(self.n);
    hw[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      hw[i] = hw[i - 1]
        + ((self.theta)(T::from_usize_(i) * dt) - self.alpha * hw[i - 1]) * dt
        + self.sigma * gn[i - 1]
    }

    hw
  }
}
