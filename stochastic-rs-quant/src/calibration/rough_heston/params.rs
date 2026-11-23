use ndarray::Array1;
use ndarray::ArrayView1;

use super::super::periodic_map;
use crate::calibration::least_squares::Room;
use crate::calibration::periodic_slopes;
use crate::calibration::stencil::room;
use crate::pricing::fourier::RoughHestonParams;

/// `(V₀, κ, θ, ν, ρ, H)`: `H`'s bounds keep it inside `(0, ½)`, `κ`'s above 0, and the model has no
/// condition coupling two of them.
pub(super) const BOX: [(f64, f64); 6] = [
  (0.005, 0.25),
  (0.01, 10.0),
  (0.001, 1.0),
  (0.01, 2.0),
  (-0.9999, 0.9999),
  (0.005, 0.495),
];

/// A raw point folded into `BOX`, coordinate by coordinate.
pub(super) fn folded(x: ArrayView1<f64>) -> RoughHestonParams {
  let places = x.iter().zip(&BOX);
  let values = places.map(|(&x, &(lo, hi))| periodic_map(x, lo, hi));
  RoughHestonParams::from(values.collect::<Array1<f64>>())
}

/// Each raw coordinate's room before its folded value meets a bound of `BOX`.
pub(super) fn room_in_box(x: ArrayView1<f64>) -> Room {
  let point = Array1::from(folded(x));
  room(&point, &BOX, periodic_slopes(x, &BOX).view())
}

impl From<RoughHestonParams> for Array1<f64> {
  fn from(p: RoughHestonParams) -> Self {
    Array1::from_vec(vec![p.v0, p.kappa, p.theta, p.nu, p.rho, p.hurst])
  }
}

impl From<Array1<f64>> for RoughHestonParams {
  fn from(v: Array1<f64>) -> Self {
    RoughHestonParams {
      v0: v[0],
      kappa: v[1],
      theta: v[2],
      nu: v[3],
      rho: v[4],
      hurst: v[5],
    }
  }
}
