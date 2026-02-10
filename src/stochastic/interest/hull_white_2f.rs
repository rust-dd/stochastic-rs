use ndarray::Array1;

use crate::stochastic::noise::cgns::CGNS;
use crate::stochastic::Float;
use crate::stochastic::Process;

/// Hull-White 2-factor model
/// dX(t) = (k(t) + U(t) - theta * X(t)) dt + sigma_1 dW1(t) x(0) = x0
/// dU(t) = b * U(t) dt + sigma_2 dW2(t) u(0) = 0
pub struct HullWhite2F<T: Float> {
  pub k: fn(T) -> T,
  pub theta: T,
  pub sigma1: T,
  pub sigma2: T,
  pub rho: T,
  pub b: T,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub n: usize,
  cgns: CGNS<T>,
}

impl<T: Float> HullWhite2F<T> {
  fn new(
    k: fn(T) -> T,
    theta: T,
    sigma1: T,
    sigma2: T,
    rho: T,
    b: T,
    x0: Option<T>,
    t: Option<T>,
    n: usize,
  ) -> Self {
    Self {
      k,
      theta,
      sigma1,
      sigma2,
      rho,
      b,
      x0,
      t,
      n,
      cgns: CGNS::new(rho, n - 1, t),
    }
  }
}

impl<T: Float> Process<T> for HullWhite2F<T> {
  type Output = [Array1<T>; 2];
  type Noise = CGNS<T>;

  fn sample(&self) -> Self::Output {
    self.euler_maruyama(|cgns| cgns.sample())
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Self::Output {
    self.euler_maruyama(|cgns| cgns.sample_simd())
  }

  fn euler_maruyama(
    &self,
    noise_fn: impl Fn(&Self::Noise) -> <Self::Noise as Process<T>>::Output,
  ) -> Self::Output {
    let dt = self.cgns.dt();
    let [cgn1, cgn2] = noise_fn(&self.cgns);

    let mut x = Array1::<T>::zeros(self.n);
    let mut u = Array1::<T>::zeros(self.n);

    x[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      x[i] = x[i - 1]
        + ((self.k)(T::from_usize(i) * dt) + u[i - 1] - self.theta * x[i - 1]) * dt
        + self.sigma1 * cgn1[i - 1];

      u[i] = u[i - 1] + self.b * u[i - 1] * dt + self.sigma2 * cgn2[i - 1];
    }

    [x, u]
  }
}
