use ndarray::Array1;

use super::cir::CIR;
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::ProcessExt;

pub struct CIR2F<T: FloatExt> {
  pub x: CIR<T>,
  pub y: CIR<T>,
  pub phi: Fn1D<T>,
}

impl<T: FloatExt> CIR2F<T> {
  pub fn new(x: CIR<T>, y: CIR<T>, phi: impl Into<Fn1D<T>>) -> Self {
    assert_eq!(x.n, y.n, "x and y CIR factors must use the same n");
    if let (Some(tx), Some(ty)) = (x.t, y.t) {
      assert!(
        (tx - ty).abs() <= T::from_f64_fast(1e-12),
        "x and y CIR factors must use the same time horizon"
      );
    }
    Self {
      x,
      y,
      phi: phi.into(),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for CIR2F<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let x = self.x.sample();
    let y = self.y.sample();

    let n = x.len();

    let dt = self.x.t.unwrap_or(T::one()) / T::from_usize_(n - 1);
    let phi = Array1::<T>::from_shape_fn(n, |i| self.phi.call(T::from_usize_(i) * dt));

    x + y + phi
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn phi_fn(t: f64) -> f64 {
    t
  }

  #[test]
  fn default_time_horizon_is_one() {
    let x = CIR::new(0.0_f64, 0.0, 0.0, 3, Some(0.0), None, Some(false));
    let y = CIR::new(0.0_f64, 0.0, 0.0, 3, Some(0.0), None, Some(false));
    let model = CIR2F::new(x, y, phi_fn as fn(f64) -> f64);

    let out = model.sample();
    assert!((out[0] - 0.0).abs() < 1e-12);
    assert!((out[1] - 0.5).abs() < 1e-12);
    assert!((out[2] - 1.0).abs() < 1e-12);
  }

  #[test]
  #[should_panic(expected = "x and y CIR factors must use the same n")]
  fn mismatched_lengths_panic() {
    let x = CIR::new(0.0_f64, 0.0, 0.0, 3, Some(0.0), Some(1.0), Some(false));
    let y = CIR::new(0.0_f64, 0.0, 0.0, 4, Some(0.0), Some(1.0), Some(false));
    let _ = CIR2F::new(x, y, phi_fn as fn(f64) -> f64);
  }
}
