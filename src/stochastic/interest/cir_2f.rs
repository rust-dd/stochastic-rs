use ndarray::Array1;

use super::cir::CIR;
use crate::f;
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

  fn sample(&self) -> Self::Output {
    let x = self.x.sample();
    let y = self.y.sample();

    let dt = self.x.t.unwrap_or(f!(0)) / f!(self.n - 1);
    let phi = Array1::<T>::from_shape_fn(self.n, |i| (self.phi)(T::from_usize(i) * dt));

    x + y * phi
  }
}
