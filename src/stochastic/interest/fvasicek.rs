use ndarray::Array1;

use crate::stochastic::diffusion::fou::FOU;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct FVasicek<T: Float> {
  pub hurst: T,
  pub theta: T,
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub fou: FOU<T>,
}

impl<T: Float> FVasicek<T> {
  pub fn new(hurst: T, theta: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
    Self {
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      fou: FOU::new(hurst, theta, mu, sigma, n, x0, t),
    }
  }
}

impl<T: Float> Process<T> for FVasicek<T> {
  type Output = Array1<T>;
  type Noise = FOU<T>;

  fn sample(&self) -> Array1<T> {
    self.fou.sample()
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Self::Output {
    self.fou.sample_simd()
  }

  fn euler_maruyama(
    &self,
    _noise_fn: impl Fn(&Self::Noise) -> <Self::Noise as Process<T>>::Output,
  ) -> Self::Output {
    unimplemented!()
  }
}
