use impl_new_derive::ImplNew;
use ndarray::Array1;

use crate::stochastic::diffusion::ou::OU;
use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

#[derive(ImplNew)]
pub struct Vasicek<T: Float> {
  pub mu: T,
  pub sigma: T,
  pub theta: Option<T>,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub ou: OU<T>,
}

impl<T: Float> Process<T> for Vasicek<T> {
  type Output = Array1<T>;
  type Noise = Gn<T>;

  fn sample(&self) -> Self::Output {
    self.ou.sample()
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Self::Output {
    self.ou.sample_simd()
  }

  fn euler_maruyama(
    &self,
    _noise_fn: impl Fn(&Self::Noise) -> <Self::Noise as Process<T>>::Output,
  ) -> Self::Output {
    unimplemented!()
  }
}
