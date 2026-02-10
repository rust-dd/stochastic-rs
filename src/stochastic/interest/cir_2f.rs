use ndarray::Array1;

use super::cir::CIR;
use crate::stochastic::noise::gn::Gn;
use crate::stochastic::Float;
use crate::stochastic::Process;

pub struct CIR2F<T: Float> {
  pub x: CIR<T>,
  pub y: CIR<T>,
  pub phi: fn(T) -> T,
}

impl<T: Float> CIR2F<T> {
  pub fn new(x: CIR<T>, y: CIR<T>, phi: fn(T) -> T, gn: Gn<T>) -> Self {
    Self { x, y, phi }
  }
}

impl<T: Float> Process<T> for CIR2F<T> {
  type Output = Array1<T>;
  type Noise = Gn<T>;

  fn sample(&self) -> Self::Output {
    let x = self.x.sample();
    let y = self.y.sample();

    let dt = self.x.t.unwrap_or(T::zero()) / T::from_usize(self.n - 1);
    let phi = Array1::<T>::from_shape_fn(self.n, |i| (self.phi)(T::from_usize(i) * dt));

    x + y * phi
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Self::Output {
    let x = self.x.sample_simd();
    let y = self.y.sample_simd();

    let dt = self.x.t.unwrap_or(T::zero()) / T::from_usize(self.n - 1);
    let phi = Array1::<T>::from_shape_fn(self.n, |i| (self.phi)(T::from_usize(i) * dt));

    x + y * phi
  }

  fn euler_maruyama(
    &self,
    _noise_fn: impl Fn(&Self::Noise) -> <Self::Noise as Process<T>>::Output,
  ) -> Self::Output {
    unimplemented!()
  }
}
