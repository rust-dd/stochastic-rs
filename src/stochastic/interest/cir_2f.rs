use ndarray::Array1;

use super::cir::CIR;
use crate::traits::Fn1D;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct CIR2F<T: FloatExt> {
  pub x: CIR<T>,
  pub y: CIR<T>,
  pub phi: Fn1D<T>,
}

impl<T: FloatExt> CIR2F<T> {
  pub fn new(x: CIR<T>, y: CIR<T>, phi: impl Into<Fn1D<T>>) -> Self {
    Self { x, y, phi: phi.into() }
  }
}

impl<T: FloatExt> ProcessExt<T> for CIR2F<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let x = self.x.sample();
    let y = self.y.sample();

    let n = x.len();

    let dt = self.x.t.unwrap_or(T::zero()) / T::from_usize_(n - 1);
    let phi = Array1::<T>::from_shape_fn(n, |i| self.phi.call(T::from_usize_(i) * dt));

    x + y + phi
  }
}
